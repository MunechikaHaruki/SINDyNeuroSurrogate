import copy
import logging

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

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


def build_dataset(
    model_name: str,
    dt: float,
    iteration: int,
    silence_steps: int,
    pipeline: list,
    current_seed: int,
    built_models: dict,
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
        "target_comp_id": built_models["target_nodes"][model_name],
        "net": built_models["mc_models"][model_name],
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
    if isinstance(catalog_item, (DictConfig, ListConfig)):
        catalog_item = OmegaConf.to_container(catalog_item, resolve=True)

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


def _build_sweep_datasets(
    sweep_name: str,
    catalog_item: dict,
    common_cfg: dict,
    built_models: dict,
) -> dict:
    """スイープ用のデータセット群を構築する"""
    datasets = {}
    model_name = catalog_item["data_type"]

    # cfgのパースと計算
    dt = common_cfg["dt"]
    duration = common_cfg["test"]["duration"]
    iteration = int(duration / dt)
    silence_steps = int(common_cfg["silence_duration"] / dt)

    # テスト用のシード値が明示されていない場合、学習用のシード値をフォールバックとして利用
    default_seed = common_cfg["test"].get(
        "current_seed", common_cfg["train"].get("current_seed")
    )

    cases = build_sweep_cases(sweep_name, catalog_item)

    for case_name, case_cfg in cases.items():
        seed = (
            case_cfg["current_seed"]
            if case_cfg["current_seed"] is not None
            else default_seed
        )
        datasets[case_name] = build_dataset(
            model_name=model_name,
            dt=dt,
            iteration=iteration,
            silence_steps=silence_steps,
            pipeline=case_cfg["pipeline"],
            current_seed=seed,
            built_models=built_models,
        )
    return datasets


def _build_train_dataset(
    catalog_item: dict,
    common_cfg: dict,
    built_models: dict,
) -> dict:
    """学習用の単一データセットを構築する"""
    model_name = catalog_item["data_type"]

    # cfgのパースと計算
    dt = common_cfg["dt"]
    duration = common_cfg["train"]["duration"]
    iteration = int(duration / dt)
    silence_steps = int(common_cfg["silence_duration"] / dt)
    default_seed = common_cfg["train"]["current_seed"]

    teach_seed = catalog_item["current"].get("current_seed", default_seed)

    return build_dataset(
        model_name=model_name,
        dt=dt,
        iteration=iteration,
        silence_steps=silence_steps,
        pipeline=catalog_item["current"]["pipeline"],
        current_seed=teach_seed,
        built_models=built_models,
    )


def build_datasets(cfg_datasets, catalog, model_definitions):
    if isinstance(cfg_datasets, DictConfig):
        cfg_datasets = OmegaConf.to_container(cfg_datasets, resolve=True)
    if isinstance(catalog, DictConfig):
        catalog = OmegaConf.to_container(catalog, resolve=True)
    if isinstance(model_definitions, DictConfig):
        model_definitions = OmegaConf.to_container(model_definitions, resolve=True)

    built_models = build_models(model_definitions)
    common_cfg = cfg_datasets["common"]

    datasets = {}

    # 1. テスト（スイープ）用データセットの構築
    sweep_name = cfg_datasets["sweep_catalog"]
    sweep_datasets = _build_sweep_datasets(
        sweep_name=sweep_name,
        catalog_item=catalog[sweep_name],
        common_cfg=common_cfg,
        built_models=built_models,
    )
    datasets.update(sweep_datasets)

    # 2. 学習用データセットの構築
    teach_name = cfg_datasets["teaching_catalog"]
    datasets["train"] = _build_train_dataset(
        catalog_item=catalog[teach_name],
        common_cfg=common_cfg,
        built_models=built_models,
    )

    logger.info(f"Built {len(datasets)} datasets (including 'train').")
    return datasets
