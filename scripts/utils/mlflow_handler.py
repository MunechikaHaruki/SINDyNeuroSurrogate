import inspect
import logging
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
import yaml
from utils.plots import (
    draw_engine,
    plot_sindy_coefficients,
    spec_simple,
)

from neurosurrogate.modeling import SINDySummary, SINDySurrogateWrapper

TARGET_EXP = "test_static_params"
mlflow.set_tracking_uri("file:./mlruns")


logger = logging.getLogger(__name__)


def setup_mlflow(is_multirun):
    mlflow.enable_system_metrics_logging()
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"
    if is_multirun:
        mlflow.set_experiment("test_dynamic_datasets")
    else:
        mlflow.set_experiment("test_static_params")


def log_surrogate_summary(summary: SINDySummary):
    mlflow.log_metrics(summary.metrics)
    mlflow.log_params(summary.params)

    for filename, content in summary.texts.items():
        mlflow.log_text(content, artifact_file=filename)

    for name, ds in summary.xarrays.items():
        datasets, spec = spec_simple(ds)
        fig = draw_engine(datasets, spec, engine="matplotlib")
        mlflow.log_figure(fig, artifact_file=f"{name}.png")

    fig = plot_sindy_coefficients(
        xi_matrix=summary.xi,
        feature_names=summary.feature_names,
        target_names=summary.target_names,
    )
    mlflow.log_figure(fig, artifact_file="sindy_coef.png")


class SINDySurrogateMLflowModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "target_module", context.artifacts["target_module_path"]
        )
        target_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(target_module)
        source = open(context.artifacts["source_path"]).read()
        local_vars = {}
        exec(source, vars(target_module), local_vars)
        compute_theta = local_vars["dynamic_compute_theta"]
        xi_matrix = np.load(context.artifacts["xi_path"])
        self.gate_init = np.load(context.artifacts["gate_init_path"])
        self.sindy_args = (xi_matrix, compute_theta)
        self.preprocessor = joblib.load(context.artifacts["preprocessor_path"])

    def predict(self, context, model_input):
        pass  # unified_simulatorに直接渡すので不要


def log_surrogate_model(surrogate: SINDySurrogateWrapper):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        np.save(tmpdir / "xi.npy", surrogate.sindy.coefficients())
        np.save(tmpdir / "gate_init.npy", surrogate.gate_init)
        (tmpdir / "source.py").write_text(surrogate.source)
        joblib.dump(surrogate.preprocessor, tmpdir / "preprocessor.joblib")

        mlflow.pyfunc.log_model(
            artifact_path="surrogate_model",
            python_model=SINDySurrogateMLflowModel(),
            artifacts={
                "xi_path": str(tmpdir / "xi.npy"),
                "gate_init_path": str(tmpdir / "gate_init.npy"),
                "source_path": str(tmpdir / "source.py"),
                "target_module_path": inspect.getfile(surrogate.target_module),
                "preprocessor_path": str(tmpdir / "preprocessor.joblib"),
            },
        )


def load_surrogate_model(run_id: str):
    # log_surrogate_model で指定した artifact_path を使用
    model_uri = f"runs:/{run_id}/surrogate_model"
    logger.info(f"Loading custom MLflow model from: {model_uri}")

    try:
        # pyfuncとしてロード（内部で load_context が実行される）
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)

        # PythonModelの実体（SINDySurrogateMLflowModelのインスタンス）を取り出す
        surrogate = pyfunc_model._model_impl.python_model

        # ログ用に属性の存在確認（任意）
        if hasattr(surrogate, "xi_matrix"):
            logger.info(f"Model loaded. Xi matrix shape: {surrogate.xi_matrix.shape}")

        return surrogate

    except Exception as e:
        logger.error(f"Failed to load surrogate from MLflow: {e}")
        raise


def get_runs_df():
    experiment = mlflow.get_experiment_by_name(TARGET_EXP)
    if experiment:
        all_runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        # 親runのみを抽出
        runs_df = all_runs_df[all_runs_df["tags.mlflow.parentRunId"].isna()].copy()
        # start_time に基づいて降順（最新が上）にソート、表示変更
        runs_df = runs_df.sort_values("start_time", ascending=False)
        runs_df["start_time"] = runs_df["start_time"].dt.strftime("%m-%d %H:%M:%S")

        # カラムの整理（run_id, start_time を先頭に）
        cols = [
            c
            for c in runs_df.columns
            if "metrics" in c or "params" in c or c == "run_id"
        ]
        runs_df = runs_df[
            ["tags.mlflow.runName", "run_id", "start_time"]
            + [c for c in cols if c != "run_id"]
        ]
        return runs_df
    else:
        return None


def get_model_informations(run_ids):
    client = mlflow.MlflowClient()
    artifact_path = "sindy_coef.png"
    download_dir = Path(tempfile.mkdtemp())
    model_info = {}
    for run_id in run_ids:
        model_info[run_id] = {}
        dest = download_dir / run_id
        dest.mkdir(exist_ok=True)

        local_path = client.download_artifacts(
            run_id=run_id, path=artifact_path, dst_path=str(dest)
        )

        model_info[run_id]["sindy_coef"] = local_path
        model_info[run_id]["runName"] = client.get_run(run_id).data.tags[
            "mlflow.runName"
        ]
        model_info[run_id]["equations"] = mlflow.artifacts.load_text(
            f"runs:/{run_id}/equations.txt"
        )
        model_info[run_id]["teaching_config"] = yaml.safe_load(
            mlflow.artifacts.load_text(f"runs:/{run_id}/dataset.yaml")
        )
    return model_info


def get_child_runs(parent_run_ids):
    # 親Runのexperiment_idを取得
    experiment_id = mlflow.get_run(parent_run_ids[0]).info.experiment_id
    # 全Runを取得してからPythonで子Runをフィルタリング
    all_runs_df = mlflow.search_runs(experiment_ids=[experiment_id])

    # 親Runに紐付く子Runをフィルタリング
    child_runs_df = all_runs_df[
        all_runs_df["tags.mlflow.parentRunId"].isin(parent_run_ids)
    ].reset_index(drop=True)
    return child_runs_df


# # 親が選択されていない場合は停止
# mo.stop(len(run_selector.value) == 0)
# with mo.status.spinner(title="子Runを取得中..."):
#     child_runs_df=get_child_runs(run_selector.value["run_id"].tolist())
# # 子Runを一覧表示（ここから特定の評価結果を選ぶ）
# if len(child_runs_df) > 0:
#     child_selector = mo.ui.table(
#         child_runs_df[["tags.mlflow.runName", "run_id", "tags.eval_dataset", "start_time"]],
#         label="Artifactを確認したい子Runを選択してください",
#         selection="single"
#     )
# else:
#     mo.md("**子Runが見つかりません**")

# child_selector
