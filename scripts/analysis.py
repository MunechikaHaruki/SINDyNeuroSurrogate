import copy
import os
from typing import get_args

import matplotlib.pyplot as plt
import mlflow
import yaml
from utils.builder import CurrentType, build_dataset, build_simulator_config
from utils.mlflow_handler import TARGET_EXP, load_surrogate_model

from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model_neurosindy import transform_gate
from neurosurrogate.profiler_view import draw_engine, spec_diff, view_model
from neurosurrogate.profiler_wave import calc_dynamic_metrics

CurrentList: list = ["train"] + list(get_args(CurrentType))


def setup_matplotlib(matplotlib_style):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    STYLE_DIR = os.path.join(CURRENT_DIR, "../conf/style")
    plt.style.use(os.path.join(STYLE_DIR, "./base.mplstyle"))
    plt.style.use(os.path.join(STYLE_DIR, f"./{matplotlib_style}.mplstyle"))


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


def get_model_infos(run_ids):

    def get_run_info(run_id: str) -> dict:
        client = mlflow.MlflowClient()

        def load_yaml(run_id: str, filename: str) -> dict:
            return yaml.safe_load(
                mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}")
            )

        def load_text(run_id: str, filename: str) -> str:
            return mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}")

        view_cfg = load_yaml(run_id, "view.json")

        return {
            "sindy_coef": view_model(**view_cfg),
            "dataset": load_yaml(
                run_id, "dataset.yaml"
            ),  # 同じファイルなら参照共有でOK
            "runName": client.get_run(run_id).data.tags["mlflow.runName"],
            "run_id": run_id,
            "equations": load_text(run_id, "equations.txt"),
        }

    model_infos = {}
    for run_id in run_ids:
        run_info = get_run_info(run_id)
        model_infos[run_id] = {}
        model_infos[run_id]["runName"] = run_info["runName"]
        model_infos[run_id]["equations"] = run_info["equations"]
        model_infos[run_id]["dataset"] = run_info["dataset"]
        model_infos[run_id]["sindy_coef"] = run_info["sindy_coef"]
    return model_infos


def resolve_config(model_infos, run_id, current_type, value):
    if current_type == "train":
        return model_infos[run_id]["dataset"]
    return build_dataset(current_type=current_type, value=value)


def eval_dataset(run_id: str, dataset_cfg: dict):

    surrogate_model = load_surrogate_model(run_id)
    built_cfg = build_simulator_config(dataset_cfg)
    net = built_cfg["net"]
    original_ds = unified_simulator(**built_cfg)
    surr_net = copy.deepcopy(net)

    target_comp_id = 0

    surr_net["nodes"][target_comp_id] = "surr"
    surr_ds = unified_simulator(
        dt=built_cfg["dt"],
        u=built_cfg["u"],
        net=surr_net,
        surrogate_model=surrogate_model,
    )
    preprocessed_xr = transform_gate(
        surrogate_model.preprocessor, original_ds, target_comp_id=target_comp_id
    )
    return {
        "datasets": {
            "original": original_ds,
            "preprocessed": preprocessed_xr,
            "surrogate": surr_ds,
            "surr_id": target_comp_id,
        },
        "metrics": calc_dynamic_metrics(
            original_ds, surr_ds, target_comp_id, dataset_cfg["dt"]
        ),
    }


# draw_engine(spec_simple(result["datasets"]["preprocessed"]))
def view_dataset(result):
    return draw_engine(spec_diff(**result["datasets"]))
