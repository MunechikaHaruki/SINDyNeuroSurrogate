import logging

import mlflow
import numpy as np
from utils.builder import build_dataset, build_simulator_config, build_surrogate
from utils.mlflow_handler import (
    load_surrogate_model,
    log_surrogate_model,
    log_surrogate_summary,
)
from utils.plots import (
    draw_engine,
    plot_2d_attractor_comparison,
    spec_diff,
)

from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model import transform_gate
from neurosurrogate.neuron_core import FUNC_COST_MAP, HH_COST
from neurosurrogate.profiler import calc_dynamic_metrics, get_loggable_summary

logger = logging.getLogger(__name__)


def cli_flow(is_multirun, cfg_sindy):
    surrogate_model = build_surrogate(cfg_sindy)
    with mlflow.start_run(run_name=f"train:{cfg_sindy['name']}") as run:
        train_run_id = run.info.run_id
        train_model(
            surrogate_model,
            build_dataset(**cfg_sindy["datasets"]),
        )
    if is_multirun:
        return eval_with_model_reaction(cfg_sindy["datasets"], train_run_id)


def train_model(surrogate, train_dataset_cfg):
    train_ds = unified_simulator(**build_simulator_config(train_dataset_cfg))
    mlflow.log_dict(train_dataset_cfg, "dataset.yaml")
    surrogate.fit(train_ds, train_dataset_cfg["target_comp_id"])
    log_surrogate_summary(get_loggable_summary(surrogate, FUNC_COST_MAP, HH_COST))
    log_surrogate_model(surrogate)


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


def eval_with_model_reaction(datasets_cfg, train_run_id):
    from scipy.signal import find_peaks

    surrogate_model = load_surrogate_model(train_run_id)

    def run_override(run_id, metric):
        current_run = mlflow.get_run(run_id)
        original_name = current_run.data.tags.get("mlflow.runName", "Run")
        mlflow.set_tag("mlflow.runName", f"{original_name} | Score:{metric:.4f}")

    @mlflow.trace
    def get_firing_count(amptitide):
        steady_cfg = build_dataset(  # あとでちゃんと実装
            dt=datasets_cfg["dt"],
            duration=datasets_cfg["duration"],
            model_name="hh",
            current_type="steady",
            value=amptitide,
        )
        surr_xr = unified_simulator(
            **build_simulator_config(steady_cfg),
            surrogate_target=steady_cfg["target_comp_id"],
            surrogate_model=surrogate_model,
        )
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
