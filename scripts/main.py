import logging
import os

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from utils.builder import (
    build_dataset,
    build_feature_cost_map,
    build_simulator_config,
    build_surrogate,
)
from utils.mlflow_handler import (
    log_surrogate_model,
    log_surrogate_summary,
    setup_mlflow,
)

from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.profiler import HH_COST, HH_RATE_COST_MAP, get_loggable_summary

# プロキシ設定を一時的に無効化
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1"


logger = logging.getLogger(__name__)


def cli_flow(is_multirun, cfg_sindy):
    surrogate = build_surrogate(cfg_sindy)
    with mlflow.start_run(run_name=f"train:{cfg_sindy['name']}"):
        # train
        train_dataset_cfg = build_dataset(**cfg_sindy["datasets"])
        train_ds = unified_simulator(**build_simulator_config(train_dataset_cfg))
        mlflow.log_dict(train_dataset_cfg, "dataset.yaml")
        train_comp_id = train_dataset_cfg["net"]["name_to_idx_dict"][
            cfg_sindy["train_comp_identifier"]
        ]
        surrogate_result = surrogate.fit(train_ds, train_comp_id)
        feature_cost = build_feature_cost_map(
            surrogate_result.base_names, HH_RATE_COST_MAP
        )
        log_surrogate_summary(
            get_loggable_summary(surrogate_result, HH_COST, feature_cost)
        )
        log_surrogate_model(surrogate)
    if is_multirun:
        pass


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    is_multirun = HydraConfig.get().mode.name == "MULTIRUN"
    setup_mlflow(is_multirun)
    cli_flow(is_multirun, cfg["sindy"])


if __name__ == "__main__":
    main()
