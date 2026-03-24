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
    log_surrogate_model,
    log_surrogate_summary,
    save_xarray,
)

from neurosurrogate.modeling.calc_engine import unified_simulator
from neurosurrogate.modeling.profiler import calc_dynamic_metrics
from neurosurrogate.utils.plots import (
    draw_engine,
    plot_2d_attractor_comparison,
    spec_diff,
)

logger = logging.getLogger(__name__)


@mlflow.trace
def train_model(surrogate, train_ds, target_comp_id):
    surrogate.fit(train_ds, target_comp_id)
    log_surrogate_summary(surrogate.get_loggable_summary())
    log_surrogate_model(surrogate)


@mlflow.trace
def eval_diff(dataset_cfg, surrogate_model):
    mlflow.log_params(dataset_cfg)
    mlflow.log_params(dataset_cfg["current"]["pipeline"][0])
    mlflow.log_dict(dataset_cfg, "dataset.yaml")

    original_ds = unified_simulator(**build_simulator_config(dataset_cfg))

    target_comp_id = dataset_cfg["target_comp_id"]

    predict_result = unified_simulator(
        **build_simulator_config(dataset_cfg),
        surrogate_target=target_comp_id,
        surrogate_model=surrogate_model,
    )

    preprocessed_xr = surrogate_model.preprocessor.transform(
        original_ds, target_comp_id=target_comp_id
    )

    # logging
    dt = dataset_cfg["dt"]
    mlflow.log_metrics(
        calc_dynamic_metrics(original_ds, predict_result, target_comp_id, dt)
    )
    names = ["orig", "preprocessed", "surr"]
    datasets = [original_ds, preprocessed_xr, predict_result]
    for ds, name in zip(datasets, names):
        save_xarray(ds, name)

    datasets, spec = spec_diff(
        original_ds, preprocessed_xr, predict_result, surr_id=target_comp_id
    )
    mlflow.log_figure(
        draw_engine(datasets, spec, engine="matplotlib"),
        artifact_file="compare.png",
    )

    fig_phase = plot_2d_attractor_comparison(
        orig_ds=preprocessed_xr,
        surr_ds=predict_result,
        comp_id=target_comp_id,
        state_vars=["V", "latent1"],  # 実際のSINDyのターゲット変数名に合わせて変更
    )
    mlflow.log_figure(fig_phase, artifact_file="attractor_surr.png")


def main_flow(datasets_cfg: Dict, surrogate_model, run_name):
    logger.info("Start Flow:start generate train data")
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"run_id:{run.info.run_id}")
        mlflow.log_dict(datasets_cfg, "datasets.yaml")
        train_ds = unified_simulator(**build_simulator_config(datasets_cfg["train"]))
        target_comp_id = datasets_cfg["train"]["target_comp_id"]
        logger.info("Start Training")
        train_model(surrogate_model, train_ds, target_comp_id)
        for key, dataset in datasets_cfg.items():
            logger.info(f"start {key}'s evaluation")
            try:
                with mlflow.start_run(run_name=f"Eval_{key}", nested=True):
                    mlflow.set_tag("eval_dataset", key)
                    eval_diff(datasets_cfg[key], surrogate_model)
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
    # 設定のビルド
    dataset_cfg = build_full_datasets(cfg, MODEL_DEFINITIONS)
    surrogate_model = build_surrogate(cfg.sindy)
    main_flow(dataset_cfg, surrogate_model, get_hydra_overrides())
    logger.info("Script ended")


if __name__ == "__main__":
    main()
