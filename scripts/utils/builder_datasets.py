import itertools
import logging

import numpy as np
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


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


def build_current_cases(current_test_settings: dict) -> list:
    """
    current_test_settings からキーと current 設定の一覧を生成する
    """
    cases = []
    base_path = "neurosurrogate.utils.current_generators."

    for current_type, spec in current_test_settings.items():
        target = base_path + spec["_target_"]
        default_params = spec.get("params", {})
        params_sweep = spec.get("params_sweep", {})

        # スイープ設定がない場合は単一のケースを作成
        if not params_sweep:
            cases.append(
                (current_type, {"pipeline": [{"_target_": target, **default_params}]})
            )
            continue

        # スイープ設定がある場合は複数ケースを展開
        sweep_key, sweep_value_cfg = _get_single_sweep_param(params_sweep)
        for val in _resolve_sweep_values(sweep_value_cfg):
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
        # ノード名からインデックスへのマップを動的に作成
        nodes_dict = spec["nodes"]
        name_to_idx = {n: i for i, n in enumerate(nodes_dict.keys())}

        mc_models[name] = {
            "nodes": list(nodes_dict.values()),
            "edges": [(name_to_idx[u], name_to_idx[v], g) for u, v, g in spec["edges"]],
            "stim_node": name_to_idx[spec["stim"]],
        }
        target_nodes[name] = name_to_idx[spec["target"]]

    return mc_models, target_nodes


def _apply_dataset_defaults(
    ds_dict: dict,
    cfg_default: dict,
    mc_models: dict,
    target_nodes: dict,
    is_train: bool,
):
    """データセットの辞書にデフォルト値を適用する（破壊的変更）"""
    dt = cfg_default["simulator_default_dt"]
    ds_dict.setdefault("dt", dt)

    current = ds_dict.setdefault("current", {})
    current.setdefault("current_seed", cfg_default["default_current_seed"])
    current.setdefault("silence_steps", int(cfg_default["silence_duration"] / dt))

    # イテレーション数の設定
    duration_key = "train_duration" if is_train else "simulator_default_duration"
    default_iteration = int(cfg_default[duration_key] / dt)
    current.setdefault("iteration", default_iteration)

    # モデル情報の紐付け
    model_name = ds_dict["data_type"]
    ds_dict["target_comp_id"] = target_nodes[model_name]
    ds_dict["net"] = mc_models[model_name]


def build_full_datasets(cfg, model_definitions):
    combo = cfg.test_combinations[cfg.active_test]
    active_models = combo["models"]
    active_currents = set(combo["currents"])  # 検索を高速化するために set に変換

    # 1. 必要な current settings だけを抽出してビルド
    raw_current_settings = OmegaConf.to_container(
        cfg.current_test_settings, resolve=True
    )
    filtered_settings = {
        k: v for k, v in raw_current_settings.items() if k in active_currents
    }
    current_cases = build_current_cases(filtered_settings)

    # 2. モデルのビルド
    mc_models, target_nodes = build_models(model_definitions)

    # 3. テストデータセットの構築 (itertools.product でネストを解消)
    datasets = {}
    for model, (case_key, current_cfg) in itertools.product(
        active_models, current_cases
    ):
        datasets[f"{case_key}_{model}"] = {
            "data_type": model,
            "current": current_cfg,
        }

    # 4. 学習用データセットの追加
    datasets["train"] = OmegaConf.to_container(
        cfg.teaching_settings[cfg.sindy.teaching_current], resolve=True
    )

    # 5. デフォルト値とモデル情報の適用
    cfg_default = cfg["datasets_default"]
    for key, ds_dict in datasets.items():
        _apply_dataset_defaults(
            ds_dict=ds_dict,
            cfg_default=cfg_default,
            mc_models=mc_models,
            target_nodes=target_nodes,
            is_train=(key == "train"),
        )

    logger.info(datasets)
    return datasets
