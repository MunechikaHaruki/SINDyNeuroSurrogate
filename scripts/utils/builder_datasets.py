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


def build_dataset(case_cfg: dict) -> dict:
    """単一のケース設定からデータセット辞書を構築する"""
    dt = case_cfg["dt"]
    model_name = case_cfg["data_type"]

    return {
        "data_type": model_name,
        "dt": dt,
        "current": {
            "current_seed": case_cfg["current_seed"],
            "iteration": int(case_cfg["duration"] / dt),
            "pipeline": case_cfg["pipeline"],
            "silence_steps": int(case_cfg["silence_duration"] / dt),
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


def build_sweep_cases(case_type: str, catalog_item: dict) -> dict:
    """カタログアイテムからベース設定を作り、スイープ展開する"""

    # 全てのベースとなる設定をまとめる
    base_cfg = {
        "data_type": catalog_item["data_type"],
        "dt": catalog_item["dt"],
        "duration": catalog_item["duration"],
        "silence_duration": catalog_item["silence_duration"],
        "current_seed": catalog_item["current"].get("current_seed", 0),
        "pipeline": catalog_item["current"]["pipeline"],
    }

    sweep_spec = catalog_item.get("sweep", {})
    cases = {}

    # 1. スイープ設定がない場合
    if not sweep_spec:
        cases[case_type] = base_cfg
        return cases

    # 2. current_seed のスイープ
    if "current_seed" in sweep_spec:
        for seed in sweep_spec["current_seed"]:
            case_cfg = copy.deepcopy(base_cfg)
            case_cfg["current_seed"] = seed
            cases[f"{case_type}_{seed}"] = case_cfg
        return cases

    # 3. pipeline 変数のスイープ
    if "current" in sweep_spec:
        c_sweep = sweep_spec["current"]
        p_idx = c_sweep.get("pipeline_ind", 0)
        variable_cfg = c_sweep.get("variable", {})

        if variable_cfg:
            sweep_key, sweep_range_cfg = _get_single_sweep_param(variable_cfg)
            sweep_values = _resolve_sweep_values(sweep_range_cfg)

            for val in sweep_values:
                case_cfg = copy.deepcopy(base_cfg)
                case_cfg["pipeline"][p_idx][sweep_key] = val
                cases[f"{case_type}_{val}"] = case_cfg
            return cases

    # フォールバック
    cases[case_type] = base_cfg
    return cases


def build_sweep_datasets(cfg_datasets) -> dict:
    """スイープ用のデータセット群を構築する"""
    catalog_name = cfg_datasets["sweep_catalog"]
    catalog_item = cfg_datasets["catalog"][catalog_name]

    # case_cfg の辞書を展開し、そのまま build_dataset に渡すだけ
    cases = build_sweep_cases(catalog_name, catalog_item)
    datasets = {name: build_dataset(cfg) for name, cfg in cases.items()}

    logger.info(f"Built {len(datasets)} datasets")
    return datasets


def build_train_dataset(cfg_datasets) -> dict:
    """学習用の単一データセットを構築する"""
    catalog_name = cfg_datasets["teaching_catalog"]
    catalog_item = cfg_datasets["catalog"][catalog_name]

    # スイープなしのベース設定をそのまま構築
    base_cfg = {
        "data_type": catalog_item["data_type"],
        "dt": catalog_item["dt"],
        "duration": catalog_item["duration"],
        "silence_duration": catalog_item["silence_duration"],
        "current_seed": catalog_item["current"].get("current_seed", 0),
        "pipeline": catalog_item["current"]["pipeline"],
    }
    return build_dataset(base_cfg)
