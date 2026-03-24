import logging
from typing import Dict

import hydra
import mlflow
from conf.neuron_models import MODEL_DEFINITIONS
from omegaconf import DictConfig
from utils.boot import setup_all
from utils.builder_core import (
    build_simulator_config,
    build_surrogate,
)
from utils.builder_datasets import build_full_datasets
from utils.log_model import log_surrogate_model
from utils.log_utils import (
    get_hydra_overrides,
    log_dataset_cfg,
    log_eval_result,
    log_surrogate_summary,
)

from neurosurrogate.modeling.calc_engine import unified_simulator

logger = logging.getLogger(__name__)


@mlflow.trace
def train_model(surrogate, train_dataset_cfg):
    target_comp_id = train_dataset_cfg["target_comp_id"]
    train_ds = unified_simulator(**build_simulator_config(train_dataset_cfg))
    with mlflow.start_run(run_name=f"Training_run:{get_hydra_overrides()}") as run:
        log_dataset_cfg(train_dataset_cfg)
        logger.info(f"run_id:{run.info.run_id}:Start training")
        surrogate.fit(train_ds, target_comp_id)
        log_surrogate_summary(surrogate.get_loggable_summary())
        log_surrogate_model(surrogate)
        return run.info.run_id


@mlflow.trace
def eval_diff(dataset_cfg, surrogate_model):
    log_dataset_cfg(dataset_cfg)
    original_ds = unified_simulator(**build_simulator_config(dataset_cfg))
    target_comp_id = dataset_cfg["target_comp_id"]
    surr_ds = unified_simulator(
        **build_simulator_config(dataset_cfg),
        surrogate_target=target_comp_id,
        surrogate_model=surrogate_model,
    )
    preprocessed_xr = surrogate_model.preprocessor.transform(
        original_ds, target_comp_id=target_comp_id
    )
    log_eval_result(original_ds, surr_ds, preprocessed_xr, dataset_cfg)


def eval_datasets(datasets_cfg: Dict, surrogate_model, train_run_id):
    logger.info("Start Flow:start generate eval data")
    for key, dataset in datasets_cfg.items():
        logger.info(f"start {key}'s evaluation")
        try:
            with mlflow.start_run(
                run_name=f"Eval_{key}",
                tags={"mlflow.parentRunId": train_run_id, "eval_dataset": key},
                nested=True,
            ):
                eval_diff(dataset, surrogate_model)
                logger.info(f"Successfully finished evaluation for {key}")
        except Exception as e:
            logger.exception(f"Failed to evaluate {key}: {str(e)}")
            mlflow.set_tag("error_type", str(type(e).__name__))
            mlflow.set_tag("error_msg", str(e))
            continue


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Activate Script")
    setup_all(cfg)
    datasets_cfg = build_full_datasets(cfg, MODEL_DEFINITIONS)
    surrogate_model = build_surrogate(cfg.sindy)
    train_run_id = train_model(surrogate_model, datasets_cfg["train"])
    eval_datasets(datasets_cfg, surrogate_model, train_run_id)
    logger.info("Script ended")


if __name__ == "__main__":
    main()
