import logging

import numpy as np
from omegaconf import OmegaConf

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


def _get_single_sweep_param(params_sweep: dict):
    if len(params_sweep) != 1:
        raise ValueError(
            f"params_sweepは1キーのみ対応しています: {list(params_sweep.keys())}"
        )
    return next(iter(params_sweep.items()))


def build_current_cases(current_test_settings):
    """
    current_test_settings からキーと current 設定の一覧を生成する
    例: [("steady_0", {...}), ("steady_10", {...}), ("random_9919", {...}), ...]
    """
    cases = []
    base_path = "neurosurrogate.utils.current_generators."

    for current_type, spec in current_test_settings.items():
        target = base_path + spec["_target_"]
        default_params = spec.get("params", {})
        params_sweep = spec.get("params_sweep", {})

        if not params_sweep:
            cases.append(
                (current_type, {"pipeline": [{"_target_": target, **default_params}]})
            )
            continue

        sweep_key, sweep_value_cfg = _get_single_sweep_param(params_sweep)
        sweep_values = _resolve_sweep_values(sweep_value_cfg)
        for val in sweep_values:
            cases.append(
                (
                    f"{current_type}_{val}",
                    {
                        "pipeline": [
                            {"_target_": target, **default_params, sweep_key: val}
                        ]
                    },
                )
            )

    return cases


def build_full_datasets(cfg, target_nodes):
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

    datasets = {}
    for model in active_models:
        target_comp_id = target_nodes[model]
        for case_key, current_cfg in current_cases:
            key = f"{case_key}_{model}"
            datasets[key] = {
                "data_type": model,
                "current": current_cfg,
                "target_comp_id": target_comp_id,
            }

    def apply_defaults(ds_dict, cfg_default, for_teaching=False):
        ds_dict.setdefault("dt", cfg_default["simulator_default_dt"])
        ds_dict["current"].setdefault(
            "current_seed", cfg_default["default_current_seed"]
        )
        ds_dict["current"].setdefault(
            "silence_steps", int(cfg_default["silence_duration"] / ds_dict["dt"])
        )
        if for_teaching is False:
            default_iteration = int(
                cfg_default["simulator_default_duration"] / ds_dict["dt"]
            )
        else:
            default_iteration = int(cfg_default["train_duration"] / ds_dict["dt"])
        ds_dict["current"].setdefault("iteration", default_iteration)
        return ds_dict

    datasets["train"] = OmegaConf.to_container(
        cfg.teaching_settings[cfg.selected], resolve=True
    )
    for key in datasets:
        datasets[key] = apply_defaults(
            datasets[key], cfg["datasets_default"], for_teaching=(key == "train")
        )

    logger.info(datasets)
    return datasets
