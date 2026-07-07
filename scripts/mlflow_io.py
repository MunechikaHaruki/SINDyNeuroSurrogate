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

from neurosurrogate.core.network import Compartment, DatasetConfig
from neurosurrogate.surrogate.neurosindy import SINDyNeuroSurrogate, make_surr_comp

TARGET_EXP = "test_static_params"


logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    project_root = Path(__file__).parent.parent
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(TARGET_EXP)
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"


SURR_ARTIFACT_DIR = "surrogate"
_XI_FILE = "xi.npy"
_GATE_INITS_FILE = "gate_inits.npy"
_SOURCE_FILE = "source.py"
_PREPROCESSOR_FILE = "preprocessor.joblib"
_MANIFEST_FILE = "manifest.json"
_DATASET_FILE = "dataset.yaml"


@dataclass(frozen=True)
class LoadedSurrogate:
    sindy_args: tuple
    preprocessor: Any
    gate_inits: list
    feature_names: list[str]
    target_names: list[str]
    equations: str
    dataset: DatasetConfig
    library_specs: list[dict]
    train_comp_identifier: str
    run_id: str
    run_name: str

    def make_surr_comp(self, name: str) -> Compartment:
        return make_surr_comp(name, self.gate_inits)

    @property
    def xi(self) -> np.ndarray:
        return self.sindy_args[0]


def log_surrogate_model(
    surrogate: SINDyNeuroSurrogate,
    dataset_cfg: DatasetConfig,
    library_specs: list[dict],
    train_comp_identifier: str,
) -> None:
    manifest = {
        "target_module": surrogate.target_module.__name__,
        "feature_names": surrogate.sindy.get_feature_names(),
        "target_names": surrogate.target_names,
        "equations": "\n".join(surrogate.sindy.equations(precision=3)),
        "library_specs": library_specs,
        "train_comp_identifier": train_comp_identifier,
    }
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        np.save(tmp / _XI_FILE, surrogate.sindy.coefficients())
        np.save(tmp / _GATE_INITS_FILE, np.array(surrogate._gate_inits))
        (tmp / _SOURCE_FILE).write_text(surrogate.source)
        joblib.dump(surrogate.preprocessor, tmp / _PREPROCESSOR_FILE)
        (tmp / _MANIFEST_FILE).write_text(json.dumps(manifest))
        (tmp / _DATASET_FILE).write_text(yaml.safe_dump(dataset_cfg.to_dict()))
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
        dataset_dict = yaml.safe_load((local / _DATASET_FILE).read_text())
        run_name = mlflow.MlflowClient().get_run(run_id).data.tags["mlflow.runName"]
        return LoadedSurrogate(
            sindy_args=(
                np.load(local / _XI_FILE),
                SINDyNeuroSurrogate._compile_source(source, target_module),
            ),
            preprocessor=joblib.load(local / _PREPROCESSOR_FILE),
            gate_inits=np.load(local / _GATE_INITS_FILE).tolist(),
            feature_names=manifest["feature_names"],
            target_names=manifest["target_names"],
            equations=manifest["equations"],
            dataset=DatasetConfig.from_dict(dataset_dict),
            library_specs=manifest["library_specs"],
            train_comp_identifier=manifest["train_comp_identifier"],
            run_id=run_id,
            run_name=run_name,
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
