import logging

import mlflow
import numpy as np
import pandas as pd
from utils.builder import build_dataset, build_simulator_config, build_surrogate
from utils.mlflow_handler import (
    load_surrogate_model,
    log_surrogate_model,
    log_surrogate_summary,
)

from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.neuron_core import FUNC_COST_MAP, HH_COST
from neurosurrogate.profiler import get_loggable_summary

logger = logging.getLogger(__name__)


def cli_flow(is_multirun, cfg_sindy):
    surrogate = build_surrogate(cfg_sindy)
    with mlflow.start_run(run_name=f"train:{cfg_sindy['name']}") as run:
        # train
        train_run_id = run.info.run_id
        train_dataset_cfg = build_dataset(**cfg_sindy["datasets"])
        train_ds = unified_simulator(**build_simulator_config(train_dataset_cfg))
        mlflow.log_dict(train_dataset_cfg, "dataset.yaml")
        surrogate.fit(train_ds, train_dataset_cfg["target_comp_id"])
        log_surrogate_summary(get_loggable_summary(surrogate, FUNC_COST_MAP, HH_COST))
        log_surrogate_model(surrogate)
    if is_multirun:
        return eval_with_model_reaction(cfg_sindy["datasets"], train_run_id)


def _format_to_table(cost_map: dict) -> str:
    # 辞書をデータフレームに変換
    df = pd.DataFrame.from_dict(cost_map, orient="index")
    df.index.name = "Feature"
    # 欠損値を0で埋めて整数型にし、美しいMarkdownとして出力
    return df.fillna(0).astype(int).to_markdown()


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
