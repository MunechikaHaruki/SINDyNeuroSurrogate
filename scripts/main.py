import logging
import os

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from mlflow_io import log_surrogate_model, setup_mlflow
from omegaconf import DictConfig, OmegaConf

from neurosurrogate.surrogate import NeuroSurrogateBase

logger = logging.getLogger(__name__)


def _disable_proxy() -> None:
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def _make_run_name() -> str:
    hc = HydraConfig.get()
    yaml_name = hc.runtime.choices["sindy"]
    extra = [
        o.rsplit(".", 1)[-1] for o in hc.overrides.task if not o.startswith("sindy=")
    ]
    return " ".join([yaml_name, *extra])


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _disable_proxy()
    setup_mlflow()
    OmegaConf.resolve(cfg)
    cfg_sindy = OmegaConf.to_container(cfg, resolve=True)["sindy"]
    assert isinstance(cfg_sindy, dict)
    surrogate = NeuroSurrogateBase.build(type=cfg_sindy["type"], init=cfg_sindy["init"])
    with mlflow.start_run(run_name=_make_run_name()):
        surrogate.fit(**cfg_sindy["fit"])
        log_surrogate_model(surrogate)


if __name__ == "__main__":
    main()
