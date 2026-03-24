import logging
import random

import hydra
import numpy as np
import pysindy as ps
from conf.feature_library_components import LIB_BUILDER_REGISTRY
from omegaconf import OmegaConf

from neurosurrogate.modeling import SINDySurrogateWrapper
from neurosurrogate.modeling.neuron_core import FUNC_COST_MAP, HH_COST

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


def build_models(definitions: dict):
    mc_models = {}
    target_nodes = {}

    for name, spec in definitions.items():
        # 1. ノード名からインデックスへのマップを動的に作成
        nodes_dict = spec["nodes"]
        node_names = list(nodes_dict.keys())
        name_to_idx = {n: i for i, n in enumerate(node_names)}

        # 2. MC_MODELS 形式の構築
        mc_models[name] = {
            "nodes": [nodes_dict[n] for n in node_names],
            "edges": [(name_to_idx[u], name_to_idx[v], g) for u, v, g in spec["edges"]],
            "stim_node": name_to_idx[spec["stim"]],
        }

        # 3. ターゲットノードのインデックス抽出
        target_nodes[name] = name_to_idx[spec["target"]]

    return mc_models, target_nodes


def build_full_datasets(cfg, model_definitions):

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
        for case_key, current_cfg in current_cases:
            key = f"{case_key}_{model}"
            datasets[key] = {
                "data_type": model,
                "current": current_cfg,
            }

    mc_models, target_nodes = build_models(model_definitions)

    def apply_defaults(ds_dict, cfg_default, for_teaching=False):
        ds_dict.setdefault("dt", cfg_default["simulator_default_dt"])
        ds_dict["current"].setdefault(
            "current_seed", cfg_default["default_current_seed"]
        )
        ds_dict["current"].setdefault(
            "silence_steps", int(cfg_default["silence_duration"] / ds_dict["dt"])
        )

        # model
        model = ds_dict["data_type"]
        ds_dict["target_comp_id"] = target_nodes[model]
        ds_dict["net"] = mc_models[model]
        if for_teaching is False:
            default_iteration = int(
                cfg_default["simulator_default_duration"] / ds_dict["dt"]
            )
        else:
            default_iteration = int(cfg_default["train_duration"] / ds_dict["dt"])
        ds_dict["current"].setdefault("iteration", default_iteration)
        return ds_dict

    datasets["train"] = OmegaConf.to_container(
        cfg.teaching_settings[cfg.sindy.teaching_current], resolve=True
    )
    for key in datasets:
        datasets[key] = apply_defaults(
            datasets[key], cfg["datasets_default"], for_teaching=(key == "train")
        )

    logger.info(datasets)
    return datasets


def build_current_pipeline(current_cfg):
    current_seed = current_cfg["current_seed"]
    iteration = current_cfg["iteration"]
    silence_steps = current_cfg["silence_steps"]
    random.seed(current_seed)
    np.random.seed(current_seed)

    dset_i_ext = np.zeros(iteration)

    for step_cfg in current_cfg["pipeline"]:
        func = hydra.utils.instantiate(step_cfg)
        func(dset_i_ext)

    dset_i_ext[:silence_steps] = 0
    dset_i_ext[-silence_steps:] = 0
    return dset_i_ext


def build_simulator_config(dataset_cfg):
    u = build_current_pipeline(dataset_cfg["current"])
    dt = dataset_cfg["dt"]
    parsed_dict = {"u": u, "dt": dt, "net": dataset_cfg["net"]}
    return parsed_dict


def build_feature_library(library_specs):

    def _build_one(spec):
        builder = LIB_BUILDER_REGISTRY.get(spec["type"])
        if builder is None:
            raise ValueError(f"未知のlibrary type: {spec['type']}")
        return builder(spec)

    lib_input_pairs = [(_build_one(s), s["inputs"]) for s in library_specs]
    libraries, inputs = zip(*lib_input_pairs)
    return ps.GeneralizedLibrary(list(libraries), inputs_per_library=list(inputs))


def build_surrogate(cfg_sindy):

    library = build_feature_library(cfg_sindy.library_specs)

    # pySINDyの初期化
    initialized_sindy = ps.SINDy(
        feature_library=library,
        optimizer=hydra.utils.instantiate(cfg_sindy.optimizer),
    )
    from neurosurrogate.modeling import neuron_core

    # surrogate_modelの初期化
    return SINDySurrogateWrapper(initialized_sindy, neuron_core, FUNC_COST_MAP, HH_COST)
