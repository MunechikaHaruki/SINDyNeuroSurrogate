import importlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure

from neurosurrogate.dataset import DatasetConfig
from neurosurrogate.metrics.sindy import SINDySummary
from neurosurrogate.registry.compartments import Compartment
from neurosurrogate.surrogate.neurosindy import SINDyNeuroSurrogate, make_surr_comp
from neurosurrogate.view.engine import view_model

TARGET_EXP = "test_static_params"


logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    project_root = Path(__file__).parent.parent
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(TARGET_EXP)
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    run_name: str
    sindy_coef: Figure
    dataset: DatasetConfig
    equations: str

    @staticmethod
    def get_run_info(run_id: str) -> "RunInfo":
        def load(filename: str) -> str:
            return mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}")

        return RunInfo(
            run_id=run_id,
            run_name=mlflow.MlflowClient().get_run(run_id).data.tags["mlflow.runName"],
            sindy_coef=view_model(**yaml.safe_load(load("view.json"))),
            dataset=DatasetConfig.from_dict(yaml.safe_load(load("dataset.yaml"))),
            equations=load("equations.txt"),
        )


def log_surrogate_summary(summary: SINDySummary):
    mlflow.log_metrics(summary.metrics)
    mlflow.log_params(summary.params)
    mlflow.log_dict(
        summary.view,
        artifact_file="view.json",
    )

    for filename, content in summary.texts.items():
        mlflow.log_text(content, artifact_file=filename)


SURR_ARTIFACT_DIR = "surrogate"
_XI_FILE = "xi.npy"
_GATE_INITS_FILE = "gate_inits.npy"
_SOURCE_FILE = "source.py"
_PREPROCESSOR_FILE = "preprocessor.joblib"
_MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class LoadedSurrogate:
    sindy_args: tuple
    preprocessor: Any
    gate_inits: list

    def make_surr_comp(self, name: str) -> Compartment:
        return make_surr_comp(name, self.gate_inits)


def log_surrogate_model(surrogate: SINDyNeuroSurrogate) -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        np.save(tmp / _XI_FILE, surrogate.sindy.coefficients())
        np.save(tmp / _GATE_INITS_FILE, np.array(surrogate._gate_inits))
        (tmp / _SOURCE_FILE).write_text(surrogate.source)
        joblib.dump(surrogate.preprocessor, tmp / _PREPROCESSOR_FILE)
        (tmp / _MANIFEST_FILE).write_text(
            json.dumps({"target_module": surrogate.target_module.__name__})
        )
        mlflow.log_artifacts(str(tmp), artifact_path=SURR_ARTIFACT_DIR)


def load_surrogate_model(run_id: str) -> LoadedSurrogate:
    logger.info(f"Loading surrogate from run {run_id}")
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{SURR_ARTIFACT_DIR}", dst_path=tmp_str
            )
        )
        manifest = json.loads((local / _MANIFEST_FILE).read_text())
        target_module = importlib.import_module(manifest["target_module"])
        source = (local / _SOURCE_FILE).read_text()
        return LoadedSurrogate(
            sindy_args=(
                np.load(local / _XI_FILE),
                SINDyNeuroSurrogate._compile_source(source, target_module),
            ),
            preprocessor=joblib.load(local / _PREPROCESSOR_FILE),
            gate_inits=np.load(local / _GATE_INITS_FILE).tolist(),
        )


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
