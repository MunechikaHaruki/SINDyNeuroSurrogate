import logging
from typing import Dict

import hydra
import mlflow
from conf.neuron_models import MODEL_DEFINITIONS
from omegaconf import DictConfig
from utils.boot import setup_all
from utils.builder import (
    build_full_datasets,
    build_simulator_config,
    build_surrogate,
)
from utils.log_utils import (
    get_hydra_overrides,
    log_dataset_cfg,
    log_eval_result,
    log_surrogate_model,
    log_surrogate_summary,
)

from neurosurrogate.modeling.calc_engine import unified_simulator

logger = logging.getLogger(__name__)


@mlflow.trace
def train_model(surrogate, train_ds, target_comp_id):
    surrogate.fit(train_ds, target_comp_id)
    log_surrogate_summary(surrogate.get_loggable_summary())
    log_surrogate_model(surrogate)


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


def main_flow(datasets_cfg: Dict, surrogate_model, run_name):
    logger.info("Start Flow:start generate train data")
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"run_id:{run.info.run_id}")
        train_ds = unified_simulator(**build_simulator_config(datasets_cfg["train"]))
        target_comp_id = datasets_cfg["train"]["target_comp_id"]
        logger.info("Start Training")
        train_model(surrogate_model, train_ds, target_comp_id)
        for key, dataset in datasets_cfg.items():
            logger.info(f"start {key}'s evaluation")
            try:
                with mlflow.start_run(run_name=f"Eval_{key}", nested=True):
                    mlflow.set_tag("eval_dataset", key)
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
    dataset_cfg = build_full_datasets(cfg, MODEL_DEFINITIONS)
    surrogate_model = build_surrogate(cfg.sindy)
    main_flow(dataset_cfg, surrogate_model, get_hydra_overrides())
    logger.info("Script ended")


if __name__ == "__main__":
    main()
