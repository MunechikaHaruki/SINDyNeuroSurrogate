from typing import get_args

import mlflow
import yaml
from utils.builder import CurrentType
from utils.mlflow_handler import TARGET_EXP
from utils.plots import plot_sindy_coefficients

CurrentList = ["train"] + list(get_args(CurrentType))


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


def get_run_info(run_id: str) -> dict:
    client = mlflow.MlflowClient()

    def load_yaml(run_id: str, filename: str) -> dict:
        return yaml.safe_load(mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}"))

    def load_text(run_id: str, filename: str) -> str:
        return mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}")

    return {
        "sindy_coef": plot_sindy_coefficients(**load_yaml(run_id, "sindy_coef.json")),
        "dataset": load_yaml(run_id, "dataset.yaml"),  # 同じファイルなら参照共有でOK
        "runName": client.get_run(run_id).data.tags["mlflow.runName"],
        "run_id": run_id,
        "equations": load_text(run_id, "equations.txt"),
    }
