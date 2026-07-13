import logging
import os
import tempfile
from pathlib import Path
from typing import cast

import mlflow
import mlflow.artifacts
import pandas as pd
import yaml

from neurosurrogate.surrogate import SURR_CLS, NeuroSurrogateBase, SurrogateMeta

TARGET_EXP = "test_static_params"

logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    project_root = Path(__file__).parent.parent
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(TARGET_EXP)
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"


SURR_ARTIFACT_DIR = "surrogate"
_META_FILE = "meta.yaml"


def log_surrogate_model(surrogate: NeuroSurrogateBase, meta: SurrogateMeta) -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        surrogate.save(tmp)
        (tmp / _META_FILE).write_text(yaml.safe_dump(meta.to_dict()))
        mlflow.log_artifacts(str(tmp), artifact_path=SURR_ARTIFACT_DIR)


def load_surrogate_model(
    run_id: str,
) -> tuple[NeuroSurrogateBase, SurrogateMeta]:
    logger.info(f"Loading surrogate from run {run_id}")
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{SURR_ARTIFACT_DIR}", dst_path=tmp_str
            )
        )
        meta = SurrogateMeta.from_dict(yaml.safe_load((local / _META_FILE).read_text()))
        surrogate = SURR_CLS[meta.surrogate_type].load(local)
    return surrogate, meta


def get_runs_df():
    experiment = mlflow.get_experiment_by_name(TARGET_EXP)
    if experiment is None:
        raise ValueError(
            f"Experiment '{TARGET_EXP}' が見つかりません。名前を確認してください。"
        )
    all_runs_df = cast(
        pd.DataFrame, mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    )
    if all_runs_df.empty:
        raise ValueError(f"Experiment '{TARGET_EXP}' にrunが存在しません。")
    runs_df = all_runs_df.copy()
    runs_df = runs_df.sort_values("start_time", ascending=False)
    runs_df["start_time"] = runs_df["start_time"].dt.strftime("%m-%d %H:%M:%S")
    runs_df = runs_df[
        ["tags.mlflow.runName", "run_id", "start_time"]
        + [c for c in runs_df.columns if "metrics" in c or "params" in c]
    ]
    return runs_df
