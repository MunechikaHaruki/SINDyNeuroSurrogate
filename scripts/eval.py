import logging
from typing import Dict

import mlflow
import numpy as np
from utils.builder_core import (
    build_simulator_config,
)
from utils.builder_datasets import (
    build_steady_dataset,
    build_sweep_datasets,
)
from utils.log_model import load_surrogate_model
from utils.log_utils import (
    _save_xarray,
    log_dataset_cfg,
    log_eval_result,
    run_override,
)

from neurosurrogate.modeling import transform_gate
from neurosurrogate.modeling.calc_engine import unified_simulator

logger = logging.getLogger(__name__)


def eval_with_static_datasets(datasets_cfg: Dict, surrogate_model, train_run_id):
    @mlflow.trace
    def eval_diff(dataset_cfg):
        log_dataset_cfg(dataset_cfg)
        original_ds = unified_simulator(**build_simulator_config(dataset_cfg))
        target_comp_id = dataset_cfg["target_comp_id"]
        surr_ds = unified_simulator(
            **build_simulator_config(dataset_cfg),
            surrogate_target=target_comp_id,
            surrogate_model=surrogate_model,
        )
        preprocessed_xr = transform_gate(
            surrogate_model.preprocessor, original_ds, target_comp_id=target_comp_id
        )
        return log_eval_result(original_ds, surr_ds, preprocessed_xr, dataset_cfg)

    logger.info("Start Flow:start generate eval data")
    datasets = build_sweep_datasets(datasets_cfg)
    for key, dataset in datasets.items():
        logger.info(f"start {key}'s evaluation")
        try:
            with mlflow.start_run(
                run_name=f"Eval_{key}",
                tags={"mlflow.parentRunId": train_run_id, "eval_dataset": key},
                nested=True,
            ) as run:
                representitive_score = eval_diff(dataset)
                run_override(run.info.run_id, representitive_score)
                logger.info(f"Successfully finished evaluation for {key}")
        except Exception as e:
            logger.exception(f"Failed to evaluate {key}: {str(e)}")
            mlflow.set_tag("error_type", str(type(e).__name__))
            mlflow.set_tag("error_msg", str(e))
            continue


def eval_with_model_reaction(datasets_cfg, train_run_id):
    from scipy.signal import find_peaks

    surrogate_model = load_surrogate_model(train_run_id)

    @mlflow.trace
    def get_firing_count(amptitide):
        steady_cfg = build_steady_dataset(datasets_cfg, amptitide)
        surr_xr = unified_simulator(
            **build_simulator_config(steady_cfg),
            surrogate_target=steady_cfg["target_comp_id"],
            surrogate_model=surrogate_model,
        )
        _save_xarray(surr_xr, "surr")
        target_data = (
            surr_xr["vars"]
            .sel(comp_id=steady_cfg["target_comp_id"], gate=False)
            .to_numpy()
            .squeeze()
        )
        peaks, _ = find_peaks(target_data, height=0.0)

        return len(peaks)

    def get_threshold():
        test_amplitudes = np.arange(0.0, 22.0, 2.0)
        for v in test_amplitudes:
            with mlflow.start_run(
                run_name=f"steady_eval_{v}",
                tags={"mlflow.parentRunId": train_run_id, "amplitude": v},
                nested=True,
            ):
                count = get_firing_count(float(v))
                mlflow.log_metric("spike_count", count)

                if count >= 10:  # 最初に5回以上発火した時の電流値を評価指標にする例
                    logger.info(f"Threshold reached at amplitude: {v}")
                    mlflow.log_metric("surr threshold", v)
                    return v
        return None

    def calc_metric(surr_threshold: float):
        if surr_threshold is None:
            return 100
        orig_threshold = (
            6.5  # これはどのモデルを採用するかによって動的に変更することに注意
        )

        return abs(orig_threshold - surr_threshold)

    v = get_threshold()
    target_metric = calc_metric(v)
    with mlflow.start_run(train_run_id):
        mlflow.log_metric("OPTUNA_TARGET_SCORE", target_metric)
        mlflow.set_tag("is_optuna_trial", "true")
        run_override(train_run_id, target_metric)
    logger.info(f"Script ended with metric: {target_metric}")
    return target_metric
