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
    例: [("steady_0", {...}), ("random_9919", {...}), ...]
    """
    cases = []

    for current_type, spec in current_test_settings.items():
        base_cfg = spec["base"]
        params_sweep = spec.get("params_sweep", {})

        # スイープ設定がない場合はベース設定をそのまま追加
        if not params_sweep:
            cases.append((current_type, {"pipeline": [base_cfg]}))
            continue

        # スイープ設定がある場合は展開してベース設定を上書き
        sweep_key, sweep_value_cfg = _get_single_sweep_param(params_sweep)
        for val in _resolve_sweep_values(sweep_value_cfg):
            # ベース設定を展開し、スイープ対象のキー(sweep_key)に値(val)をセット
            pipeline_item = {**base_cfg, sweep_key: val}
            cases.append(
                (
                    f"{current_type}_{val}",
                    {"pipeline": [pipeline_item]},
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


def build_dataset(
    model_name: str,
    current_cfg: dict,
    cfg_default: dict,
    mc_models: dict,
    target_nodes: dict,
    is_train: bool = False,
) -> dict:
    """
    単一のデータセット構成を作成し、デフォルト値とモデル情報を適用して返す。
    """
    # 基礎構造の作成
    ds_dict = {
        "data_type": model_name,
        "current": current_cfg,
    }

    # デフォルト値の適用 (既存のロジックを流用)
    dt = cfg_default["simulator_default_dt"]
    ds_dict.setdefault("dt", dt)

    current = ds_dict["current"]  # 参照を取得
    current.setdefault("current_seed", cfg_default["default_current_seed"])
    current.setdefault("silence_steps", int(cfg_default["silence_duration"] / dt))

    # イテレーション数の設定
    duration_key = "train_duration" if is_train else "simulator_default_duration"
    default_iteration = int(cfg_default[duration_key] / dt)
    current.setdefault("iteration", default_iteration)

    # モデル情報の紐付け
    ds_dict["target_comp_id"] = target_nodes[model_name]
    ds_dict["net"] = mc_models[model_name]

    return ds_dict


def build_train_dataset(cfg, mc_models, target_nodes) -> dict:
    """学習用データセットを構築する"""
    cfg_default = cfg["datasets_default"]

    # 設定の取得
    train_raw = OmegaConf.to_container(
        cfg.teaching_settings[cfg.sindy.teaching_current], resolve=True
    )

    # 共通の構築ロジックを呼び出す
    return build_dataset(
        model_name=train_raw["data_type"],
        current_cfg=train_raw["current"],
        cfg_default=cfg_default,
        mc_models=mc_models,
        target_nodes=target_nodes,
        is_train=True,
    )


def build_full_datasets(cfg, model_definitions):
    # 1. 共通リソースの準備
    mc_models, target_nodes = build_models(model_definitions)
    cfg_default = cfg["datasets_default"]

    # 2. 学習用データセットの生成 (切り出し)
    datasets = {"train": build_train_dataset(cfg, mc_models, target_nodes)}

    # 3. テスト用 Current Cases の生成
    combo = cfg.test_combinations[cfg.active_test]
    active_currents = set(combo["currents"])
    raw_current_settings = OmegaConf.to_container(
        cfg.current_test_settings, resolve=True
    )
    filtered_settings = {
        k: v for k, v in raw_current_settings.items() if k in active_currents
    }
    current_cases = build_current_cases(filtered_settings)

    # 4. テストデータセットの構築 (ループ)
    active_models = combo["models"]
    for model, (case_key, current_cfg) in itertools.product(
        active_models, current_cases
    ):
        datasets[f"{case_key}_{model}"] = build_dataset(
            model_name=model,
            current_cfg=current_cfg,
            cfg_default=cfg_default,
            mc_models=mc_models,
            target_nodes=target_nodes,
            is_train=False,
        )

    logger.info(f"Built {len(datasets)} datasets (including 'train').")
    return datasets
