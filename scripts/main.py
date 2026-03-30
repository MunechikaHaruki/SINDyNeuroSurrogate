import logging

import hydra
import mlflow
from eval import eval_with_model_reaction, eval_with_static_datasets
from omegaconf import DictConfig
from utils.boot import setup_all
from utils.builder_core import (
    build_simulator_config,
    build_surrogate,
)
from utils.builder_datasets import (
    build_train_dataset,
)
from utils.log_model import log_surrogate_model
from utils.log_utils import (
    get_hydra_overrides,
    log_dataset_cfg,
    log_surrogate_summary,
)

from neurosurrogate.modeling.calc_engine import unified_simulator

logger = logging.getLogger(__name__)


@mlflow.trace
def train_model(surrogate, train_dataset_cfg):
    train_ds = unified_simulator(**build_simulator_config(train_dataset_cfg))
    log_dataset_cfg(train_dataset_cfg)
    surrogate.fit(train_ds, train_dataset_cfg["target_comp_id"])
    log_surrogate_summary(surrogate.get_loggable_summary())
    log_surrogate_model(surrogate)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Activate Script")
    cfg = setup_all(cfg)
    surrogate_model = build_surrogate(cfg["sindy"])

    datasets_cfg = cfg["datasets_settings"]
    with mlflow.start_run(run_name=f"Training_run:{get_hydra_overrides()}") as run:
        train_run_id = run.info.run_id
        logger.info(f"run_id:{run.info.run_id}:Start training")
        train_model(surrogate_model, build_train_dataset(datasets_cfg))

    if cfg["is_multirun"]:
        return eval_with_model_reaction(datasets_cfg, train_run_id)
    else:
        eval_with_static_datasets(datasets_cfg, surrogate_model, train_run_id)


if __name__ == "__main__":
    main()
