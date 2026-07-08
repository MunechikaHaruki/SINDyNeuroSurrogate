import logging
import os

import hydra
import mlflow
from mlflow_io import log_surrogate_model, setup_mlflow
from omegaconf import DictConfig, OmegaConf

from neurosurrogate.core.network import DatasetConfig
from neurosurrogate.core.simulator import unified_simulator
from neurosurrogate.registry.neuron import MCMODELS
from neurosurrogate.surrogate.neurosindy import (
    HybridSINDyNeuroSurrogate,
    SINDyNeuroSurrogate,
)

logger = logging.getLogger(__name__)


def _disable_proxy() -> None:
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def cli_flow(cfg_sindy):
    surr_cls = (
        HybridSINDyNeuroSurrogate
        if cfg_sindy.get("hybrid", False)
        else SINDyNeuroSurrogate
    )
    surrogate = surr_cls(
        preprocessor=hydra.utils.instantiate(cfg_sindy["preprocessor"]),
        optimizer=hydra.utils.instantiate(cfg_sindy["optimizer"]),
        library_specs=cfg_sindy["library_specs"],
    )
    with mlflow.start_run(run_name=f"{cfg_sindy['name']}"):
        train_dataset_cfg = DatasetConfig.build_dataset(**cfg_sindy["datasets"])
        surrogate.fit(
            unified_simulator(train_dataset_cfg),
            MCMODELS[cfg_sindy["datasets"]["model_name"]].name_to_idx(
                cfg_sindy["train_comp_identifier"]
            ),
        )
        log_surrogate_model(surrogate, train_dataset_cfg)


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
