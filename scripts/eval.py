import logging
from typing import Dict

import mlflow
import numpy as np
from utils.builder import (
    build_simulator_config,
    build_steady_dataset,
    build_sweep_datasets,
)
from utils.log_model import (
    _save_xarray,
    load_surrogate_model,
)

from neurosurrogate.modeling import transform_gate
from neurosurrogate.modeling.calc_engine import unified_simulator
from neurosurrogate.modeling.profiler import calc_dynamic_metrics
from neurosurrogate.utils.plots import (
    draw_engine,
    plot_2d_attractor_comparison,
    spec_diff,
)

logger = logging.getLogger(__name__)


def _log_eval_result(original_ds, surr_ds, preprocessed_xr, dataset_cfg):
    target_comp_id = dataset_cfg["target_comp_id"]
    metrics = calc_dynamic_metrics(
        original_ds, surr_ds, target_comp_id, dataset_cfg["dt"]
    )
    datasets, spec = spec_diff(
        original_ds, preprocessed_xr, surr_ds, surr_id=target_comp_id
    )
    fig_diff = draw_engine(datasets, spec, engine="matplotlib")
    fig_phase = plot_2d_attractor_comparison(
        orig_ds=preprocessed_xr,
        surr_ds=surr_ds,
        comp_id=target_comp_id,
        state_vars=["V", "latent1"],  # 実際のSINDyのターゲット変数名に合わせて変更
    )
    mlflow.log_figure(fig_phase, artifact_file="attractor_surr.png")
    return {"figure": {"diff": fig_diff, "phase": fig_phase}, "metrics": metrics}


def eval_dataset(surrogate_model, dataset_cfg):
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
    return _log_eval_result(original_ds, surr_ds, preprocessed_xr, dataset_cfg)


def eval_datasets(datasets_cfg: Dict, surrogate_model):
    datasets = build_sweep_datasets(datasets_cfg)
    for key, dataset_cfg in datasets.items():
        eval_dataset(surrogate_model, dataset_cfg)


def run_override(run_id, metric):
    current_run = mlflow.get_run(run_id)
    original_name = current_run.data.tags.get("mlflow.runName", "Run")
    mlflow.set_tag("mlflow.runName", f"{original_name} | Score:{metric:.4f}")


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
