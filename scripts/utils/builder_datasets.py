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
    """単一のケース設定(YAMLのcatalog_itemの階層構造そのまま)からデータセット辞書を構築する"""
    dt = case_cfg["dt"]
    model_name = case_cfg["data_type"]

    return {
        "data_type": model_name,
        "dt": dt,
        "current": {
            # フラットアクセスではなく、case_cfg["current"] のネストを参照する
            "current_seed": case_cfg["current"]["current_seed"],
            "iteration": int(case_cfg["duration"] / dt),
            "pipeline": case_cfg["current"]["pipeline"],
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
    """カタログアイテムをベースに、スイープ条件のみを差し替えた設定群を生成する"""

    # 1. ベース設定の作成（分解せず、まるごとコピー）
    # catalog_item の中身がそのまま base になるため、フィールドが増えても修正不要
    base_cfg = copy.deepcopy(catalog_item)

    # 不要な sweep 定義は、個別のケース設定からは取り除いておく（クリーンにするため）
    sweep_spec = base_cfg.pop("sweep", {})
    cases = {}

    # 2. スイープ設定がない場合はそのまま返す
    if not sweep_spec:
        cases[case_type] = base_cfg
        return cases

    # 3. current_seed のスイープ (最上位階層の上書き)
    if "current_seed" in sweep_spec:
        for seed in sweep_spec["current_seed"]:
            case_cfg = copy.deepcopy(base_cfg)
            case_cfg["current"]["current_seed"] = seed  # ネストに直接代入
            cases[f"{case_type}_{seed}"] = case_cfg
        return cases

    # 4. pipeline 変数のスイープ (深い階層の上書き)
    if "current" in sweep_spec:
        c_sweep = sweep_spec["current"]
        p_idx = c_sweep.get("pipeline_ind", 0)
        variable_cfg = c_sweep.get("variable", {})

        if variable_cfg:
            sweep_key, sweep_range_cfg = _get_single_sweep_param(variable_cfg)
            sweep_values = _resolve_sweep_values(sweep_range_cfg)

            for val in sweep_values:
                case_cfg = copy.deepcopy(base_cfg)
                # 決め打ちの分解ではなく、構造を維持したまま値を差し替える
                case_cfg["current"]["pipeline"][p_idx][sweep_key] = val
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
    case_cfg = copy.deepcopy(catalog_item)
    case_cfg.pop("sweep", None)

    return build_dataset(case_cfg)


def build_steady_dataset(cfg_datasets: dict, amplitude: float) -> dict:
    """
    catalog['steady'] をベースに、指定された強度の定常電流データセット設定を生成する。

    Args:
        cfg_datasets: datasets_settings 全体の辞書
        amplitude: 注入する定常電流の強度
    """
    catalog_item = cfg_datasets["catalog"]["steady"]
    case_cfg = copy.deepcopy(catalog_item)
    case_cfg["current"]["pipeline"][0]["value"] = amplitude
    case_cfg.pop("sweep", None)
    return build_dataset(case_cfg)
