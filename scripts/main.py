import logging
import os

import hydra
import mlflow
import pysindy as ps
from io_handler import (
    log_surrogate_model,
    log_surrogate_summary,
    setup_mlflow,
)
from omegaconf import DictConfig, OmegaConf

from neurosurrogate.builder import registry_feature_libraries
from neurosurrogate.builder.builder_feature_libraries import FeatureLibrary
from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model.model_dataset import DatasetConfig
from neurosurrogate.model.model_neurosindy import SINDyNeuroSurrogate
from neurosurrogate.model.registry_compartments import COMPARTMENT_TEMPLATES
from neurosurrogate.model.registry_neuron import MCMODELS
from neurosurrogate.profiler.profiler_model import SINDyAnalyzer

# プロキシ設定を一時的に無効化
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1"


logger = logging.getLogger(__name__)
setup_mlflow()


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
        surrogate_result = surrogate.fit(
            unified_simulator(
                dt=train_dataset_cfg.dt,
                u=train_dataset_cfg.current.build(),
                net=train_dataset_cfg.net,
            ),
            MCMODELS[cfg_sindy["datasets"]["model_name"]].name_to_idx(
                cfg_sindy["train_comp_identifier"]
            ),
        )
        log_surrogate_summary(
            SINDyAnalyzer(
                surrogate_result,
                feature_lib.to_base_cost(surrogate_result.feature_names_in),
                original_cost=COMPARTMENT_TEMPLATES["hh"].OpCost,
            )
        )
        log_surrogate_model(surrogate)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    cli_flow(cfg_dict["sindy"])


if __name__ == "__main__":
    main()
