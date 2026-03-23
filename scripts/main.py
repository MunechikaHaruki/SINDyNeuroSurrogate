import logging
import os

import hydra
import matplotlib
import mlflow
import numpy as np
from base import COST_MAP, MC_MODELS, SINDY_MODEl
from flow import main_flow
from omegaconf import DictConfig, OmegaConf

from neurosurrogate.modeling import SINDySurrogateWrapper

matplotlib.use("Agg")


import matplotlib.pyplot as plt

# プロキシ設定を一時的に無効化
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1"


logger = logging.getLogger(__name__)


def _resolve_sweep_values(value_cfg) -> list:
    if isinstance(value_cfg, list):
        return value_cfg
    if isinstance(value_cfg, dict):
        start = value_cfg["start"]
        stop = value_cfg["stop"]
        if "num" in value_cfg:
            return np.linspace(start, stop, value_cfg["num"]).tolist()
        elif "step" in value_cfg:
            return np.arange(
                start, stop + value_cfg["step"], value_cfg["step"]
            ).tolist()
    raise ValueError(f"params_sweepの値が不正です: {value_cfg}")


def build_current_cases(current_test_settings):
    """
    current_test_settings からキーと current 設定の一覧を生成する
    例: [("steady_0", {...}), ("steady_10", {...}), ("random_9919", {...}), ...]
    """
    cases = []
    for current_type, spec in current_test_settings.items():
        base_path = "neurosurrogate.utils.current_generators."
        target = spec["_target_"]
        target = base_path + target
        default_params = spec.get("params", {})
        params_sweep = spec.get("params_sweep", {})

        if not params_sweep:
            cases.append((current_type, {"_target_": target, **default_params}))
            continue

        sweep_key, sweep_value_cfg = next(iter(params_sweep.items()))
        sweep_values = _resolve_sweep_values(sweep_value_cfg)
        for val in sweep_values:
            current_cfg = {"_target_": target, **default_params, sweep_key: val}
            cases.append((f"{current_type}_{val}", current_cfg))
    return cases


def build_full_datasets(cfg):
    datasets = {"train": OmegaConf.to_container(cfg.train, resolve=True)}
    combo = cfg.test_combinations[cfg.active_test]
    active_models = combo["models"]
    active_currents = combo["currents"]

    current_cases = build_current_cases(
        {
            k: v
            for k, v in OmegaConf.to_container(
                cfg.current_test_settings, resolve=True
            ).items()
            if k in active_currents
        }
    )

    for model in active_models:
        target_comp_id = SINDY_MODEl["target"][model]
        for case_key, current_cfg in current_cases:
            key = f"{case_key}_{model}"
            datasets[key] = {
                "data_type": model,
                "current": current_cfg,
                "target_comp_id": target_comp_id,
            }

    def apply_defaults(ds_dict, cfg_default):
        ds_dict.setdefault("dt", cfg_default["simulator_default_dt"])
        ds_dict["current"].setdefault(
            "current_seed", cfg_default["default_current_seed"]
        )
        ds_dict["current"].setdefault(
            "silence_steps", int(cfg_default["silence_duration"] / ds_dict["dt"])
        )
        ds_dict["current"].setdefault(
            "iteration", int(cfg_default["simulator_default_duration"] / ds_dict["dt"])
        )
        return ds_dict

    for key in datasets:
        datasets[key] = apply_defaults(datasets[key], cfg["datasets_default"])
    logger.info(datasets)
    return datasets


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Activate Script")
    # cfgの依存関係解決とビルド
    OmegaConf.resolve(cfg)
    dataset_cfg = build_full_datasets(cfg)
    logger.info(dataset_cfg)
    # mlflowの初期設定
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(cfg.experiment_name)
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"
    # matplotlibのstyle設定
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_STYLE_PATH = os.path.join(BASE_DIR, "./conf/style/base.mplstyle")
    plt.style.use(BASE_STYLE_PATH)
    STYLE_PATH = os.path.join(BASE_DIR, f"./conf/style/{cfg.matplotlib_style}.mplstyle")
    plt.style.use(STYLE_PATH)

    # surrogate_modelの初期化
    surrogate_model = SINDySurrogateWrapper(
        SINDY_MODEl["sindy"], SINDY_MODEl["env"], COST_MAP["func"], COST_MAP["orig"]
    )
    # mlflow name
    try:
        hydra_overrides = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
    except Exception:
        hydra_overrides = "OverrideError"
    if hydra_overrides == "":
        hydra_overrides = "Default"
    # Prefect flow
    main_flow(dataset_cfg, surrogate_model, MC_MODELS, hydra_overrides)
    logger.info("Script ended")


if __name__ == "__main__":
    main()
