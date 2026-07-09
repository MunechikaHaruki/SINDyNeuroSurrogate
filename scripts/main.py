import logging
import os

import hydra
import mlflow
from mlflow_io import log_surrogate_model, setup_mlflow
from omegaconf import DictConfig, OmegaConf

from neurosurrogate.registry.neurosindy import (
    HybridSINDyNeuroSurrogate,
    SINDyNeuroSurrogate,
)

logger = logging.getLogger(__name__)

_SURR_CLS = {
    "sindy": SINDyNeuroSurrogate,
    "hybrid": HybridSINDyNeuroSurrogate,
}


def _disable_proxy() -> None:
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _disable_proxy()
    setup_mlflow()
    OmegaConf.resolve(cfg)
    cfg_sindy = OmegaConf.to_container(cfg, resolve=True)["sindy"]
    assert isinstance(cfg_sindy, dict)
    surrogate = _SURR_CLS[cfg_sindy.get("type", "sindy")](**cfg_sindy["init"])
    with mlflow.start_run(run_name=cfg_sindy["name"]):
        surrogate.fit(**cfg_sindy["fit"])
        log_surrogate_model(surrogate, surrogate._dataset)


if __name__ == "__main__":
    main()
