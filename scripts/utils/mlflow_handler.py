import inspect
import logging
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
import yaml

from neurosurrogate.model import SINDyNeuroSurrogate
from neurosurrogate.profiler import SINDySummary

TARGET_EXP = "test_static_params"


logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent  # 階層に応じて調整
mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")
mlflow.enable_system_metrics_logging()
os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"


def setup_mlflow(is_multirun):
    if is_multirun:
        mlflow.set_experiment("test_dynamic_datasets")
    else:
        mlflow.set_experiment("test_static_params")


def log_surrogate_summary(summary: SINDySummary):
    mlflow.log_metrics(summary.metrics)
    mlflow.log_params(summary.params)

    for filename, content in summary.texts.items():
        mlflow.log_text(content, artifact_file=filename)

    mlflow.log_dict(
        {
            "xi_matrix": summary.xi.tolist(),  # numpy → list
            "feature_names": summary.feature_names,
            "target_names": summary.target_names,
        },
        artifact_file="sindy_coef.json",
    )


def get_run_info(run_id: str) -> dict:
    client = mlflow.MlflowClient()

    def load_yaml(run_id: str, filename: str) -> dict:
        return yaml.safe_load(mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}"))

    def load_text(run_id: str, filename: str) -> str:
        return mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}")

    return {
        "sindy_coef": load_yaml(run_id, "sindy_coef.json"),
        "dataset": load_yaml(run_id, "dataset.yaml"),  # 同じファイルなら参照共有でOK
        "runName": client.get_run(run_id).data.tags["mlflow.runName"],
        "run_id": run_id,
        "equations": load_text(run_id, "equations.txt"),
    }


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


def log_surrogate_model(surrogate: SINDyNeuroSurrogate):
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
    if experiment is None:
        raise ValueError(
            f"Experiment '{TARGET_EXP}' が見つかりません。名前を確認してください。"
        )
    all_runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if all_runs_df.empty:
        raise ValueError(f"Experiment '{TARGET_EXP}' にrunが存在しません。")
    runs_df = all_runs_df.copy()
    runs_df = runs_df.sort_values("start_time", ascending=False)
    runs_df["start_time"] = runs_df["start_time"].dt.strftime("%m-%d %H:%M:%S")
    cols = [
        c for c in runs_df.columns if "metrics" in c or "params" in c or c == "run_id"
    ]
    runs_df = runs_df[
        ["tags.mlflow.runName", "run_id", "start_time"]
        + [c for c in cols if c != "run_id"]
    ]
    return runs_df


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
