import logging
import os

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from mlflow_io import log_surrogate_model, setup_mlflow
from omegaconf import DictConfig, OmegaConf

from neurosurrogate.surrogate.bundle import SurrogateBundle

logger = logging.getLogger(__name__)


def _disable_proxy() -> None:
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def _preset_name() -> str:
    """選択された surrogate preset (surrogate/*.yaml) 名。"""
    return str(HydraConfig.get().runtime.choices["surrogate"])


def _make_run_name() -> str:
    extra = [
        o.rsplit(".", 1)[-1]
        for o in HydraConfig.get().overrides.task
        if not o.startswith("surrogate=")
    ]
    return " ".join([_preset_name(), *extra])


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    _disable_proxy()
    setup_mlflow()
    OmegaConf.resolve(cfg)
    cfg_surr = OmegaConf.to_container(cfg, resolve=True)["surrogate"]
    assert isinstance(cfg_surr, dict)
    run_name = _make_run_name()
    logger.info(f"[{run_name}] fit 開始")
    with mlflow.start_run(run_name=run_name):
        # 出自 (どの yaml から来たか) は学習結果に影響しないので surrogate の pickle
        # でなく実験メタデータ側へ。MLflow UI でも marimo でも同じキーで絞れる。
        mlflow.log_param("preset", _preset_name())
        surrogate = SurrogateBundle.setup(cfg_surr)
        log_surrogate_model(surrogate)
        logger.info(f"[{run_name}] 完了")


if __name__ == "__main__":
    main()
