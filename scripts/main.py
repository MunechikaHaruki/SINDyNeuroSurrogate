import logging
import os

import hydra
import mlflow
import pysindy as ps
from mlflow_io import (
    log_surrogate_model,
    log_surrogate_summary,
    setup_mlflow,
)
from omegaconf import DictConfig, OmegaConf

from neurosurrogate.core.network import DatasetConfig
from neurosurrogate.core.simulator import unified_simulator
from neurosurrogate.surrogate.libraries import FeatureLibrary
from neurosurrogate.surrogate.analysis import sindy_analysis
from neurosurrogate.registry import feature_libraries as registry_feature_libraries
from neurosurrogate.registry.compartments import COMPARTMENT_TEMPLATES
from neurosurrogate.registry.neuron import MCMODELS
from neurosurrogate.surrogate.neurosindy import SINDyNeuroSurrogate

logger = logging.getLogger(__name__)


def _disable_proxy() -> None:
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def _build_surrogate(cfg_sindy):
    feature_lib = FeatureLibrary.build(cfg_sindy["library_specs"])
    return SINDyNeuroSurrogate(
        hydra.utils.instantiate(cfg_sindy["preprocessor"]),
        ps.SINDy(
            feature_library=feature_lib.library,
            optimizer=hydra.utils.instantiate(cfg_sindy["optimizer"]),
        ),
        registry_feature_libraries,
    ), feature_lib


def cli_flow(cfg_sindy):
    surrogate, feature_lib = _build_surrogate(cfg_sindy)
    with mlflow.start_run(run_name=f"{cfg_sindy['name']}"):
        # train
        train_dataset_cfg = DatasetConfig.build_dataset(**cfg_sindy["datasets"])

        mlflow.log_dict(train_dataset_cfg.to_dict(), "dataset.yaml")
        surrogate.fit(
            unified_simulator(train_dataset_cfg),
            MCMODELS[cfg_sindy["datasets"]["model_name"]].name_to_idx(
                cfg_sindy["train_comp_identifier"]
            ),
        )
        log_surrogate_summary(
            sindy_analysis(
                surrogate,
                feature_cost_map=feature_lib.to_base_cost(
                    surrogate.sindy.feature_names
                ),
                original_cost=COMPARTMENT_TEMPLATES["hh"].OpCost,
            )
        )
        log_surrogate_model(surrogate)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _disable_proxy()
    setup_mlflow()
    OmegaConf.resolve(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    cli_flow(cfg_dict["sindy"])


if __name__ == "__main__":
    main()
