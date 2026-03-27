import copy
import logging

import numpy as np
from conf.neuron_models import MODEL_DEFINITIONS

logger = logging.getLogger(__name__)


def build_models(definitions: dict):
    mc_models = {}
    target_nodes = {}

    for name, spec in definitions.items():
        nodes_dict = spec["nodes"]
        name_to_idx = {n: i for i, n in enumerate(nodes_dict.keys())}

        mc_models[name] = {
            "nodes": list(nodes_dict.values()),
            "edges": [(name_to_idx[u], name_to_idx[v], g) for u, v, g in spec["edges"]],
            "stim_node": name_to_idx[spec["stim"]],
        }
        target_nodes[name] = name_to_idx[spec["target"]]

    return {"mc_models": mc_models, "target_nodes": target_nodes}


BUILT_MODELS = build_models(MODEL_DEFINITIONS)


def build_dataset(
    model_name: str,
    dt: float,
    iteration: int,
    silence_steps: int,
    pipeline: list,
    current_seed: int,
) -> dict:
    return {
        "data_type": model_name,
        "dt": dt,
        "current": {
            "current_seed": current_seed,
            "iteration": iteration,
            "pipeline": pipeline,
            "silence_steps": silence_steps,
        },
        "target_comp_id": BUILT_MODELS["target_nodes"][model_name],
        "net": BUILT_MODELS["mc_models"][model_name],
    }


def _get_single_sweep_param(params_sweep: dict):
    if len(params_sweep) != 1:
        raise ValueError(
            f"params_sweepは1キーのみ対応しています: {list(params_sweep.keys())}"
        )
    return next(iter(params_sweep.items()))


def _resolve_sweep_values(value_cfg) -> list:
    if isinstance(value_cfg, list):
        return value_cfg
    if isinstance(value_cfg, dict):
        start, stop = value_cfg["start"], value_cfg["stop"]
        if "num" in value_cfg:
            return np.linspace(start, stop, value_cfg["num"]).tolist()
        if "step" in value_cfg:
            # Note: np.arangeのstopは含まれないため + step をしている仕様を維持
            return np.arange(
                start, stop + value_cfg["step"], value_cfg["step"]
            ).tolist()
    raise ValueError(f"params_sweepの値が不正です: {value_cfg}")


def build_sweep_cases(case_type: str, catalog_item) -> dict:
    """
    カタログスペックからスイープ設定を展開し、ケースごとの設定辞書を返す。
    Returns: { "case_name": {"pipeline": [...], "current_seed": ...}, ... }
    """
    base_pipeline = catalog_item["current"]["pipeline"]
    base_seed = catalog_item["current"].get("current_seed")
    sweep_spec = catalog_item.get("sweep", {})

    cases = {}

    # 1. スイープ設定がない場合
    if not sweep_spec:
        cases[case_type] = {"pipeline": base_pipeline, "current_seed": base_seed}
        return cases

    # 2. current_seed のスイープ (random 用)
    if "current_seed" in sweep_spec:
        for seed in sweep_spec["current_seed"]:
            cases[f"{case_type}_{seed}"] = {
                "pipeline": copy.deepcopy(base_pipeline),
                "current_seed": seed,
            }
        return cases

    # 3. pipeline 変数のスイープ (steady 用)
    if "current" in sweep_spec:
        c_sweep = sweep_spec["current"]
        p_idx = c_sweep.get("pipeline_ind", 0)
        variable_cfg = c_sweep.get("variable", {})

        if variable_cfg:
            sweep_key, sweep_range_cfg = _get_single_sweep_param(variable_cfg)
            sweep_values = _resolve_sweep_values(sweep_range_cfg)

            for val in sweep_values:
                new_pipeline = copy.deepcopy(base_pipeline)
                new_pipeline[p_idx][sweep_key] = val

                cases[f"{case_type}_{val}"] = {
                    "pipeline": new_pipeline,
                    "current_seed": base_seed,
                }
            return cases

    # どの条件にも合致しない場合はフォールバック
    cases[case_type] = {"pipeline": base_pipeline, "current_seed": base_seed}
    return cases


def build_sweep_datasets(cfg_datasets) -> dict:
    """スイープ用のデータセット群を構築する"""

    catalog_name = cfg_datasets["sweep_catalog"]
    catalog_item = cfg_datasets["catalog"][catalog_name]
    common_cfg = cfg_datasets["common"]

    cases = build_sweep_cases(catalog_name, catalog_item)
    dt = common_cfg["dt"]
    datasets = {}
    for case_name, case_cfg in cases.items():
        seed = (
            case_cfg["current_seed"]
            if case_cfg["current_seed"] is not None
            else common_cfg["test"]["current_seed"]
        )
        datasets[case_name] = build_dataset(
            model_name=catalog_item["data_type"],
            dt=dt,
            iteration=int(common_cfg["test"]["duration"] / dt),
            silence_steps=int(common_cfg["silence_duration"] / dt),
            pipeline=case_cfg["pipeline"],
            current_seed=seed,
        )
    logger.info(f"Built {len(datasets)} datasets")
    return datasets


def build_train_dataset(cfg_datasets) -> dict:
    """学習用の単一データセットを構築する"""
    catalog_name = cfg_datasets["sweep_catalog"]
    catalog_item = cfg_datasets["catalog"][catalog_name]
    common_cfg = cfg_datasets["common"]

    model_name = catalog_item["data_type"]
    teach_seed = catalog_item["current"].get(
        "current_seed", common_cfg["train"]["current_seed"]
    )
    dt = common_cfg["dt"]
    return build_dataset(
        model_name=model_name,
        dt=dt,
        iteration=int(common_cfg["train"]["duration"] / dt),
        silence_steps=int(common_cfg["silence_duration"] / dt),
        pipeline=catalog_item["current"]["pipeline"],
        current_seed=teach_seed,
    )
