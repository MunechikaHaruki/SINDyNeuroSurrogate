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
    dt, silence_duration, duration, current_seed, model_name, pipeline
) -> dict:
    """単一のケース設定(YAMLのcatalog_itemの階層構造そのまま)からデータセット辞書を構築する"""
    return {
        "data_type": model_name,
        "dt": dt,
        "current": {
            # フラットアクセスではなく、case_cfg["current"] のネストを参照する
            "current_seed": current_seed,
            "iteration": int(duration / dt),
            "pipeline": pipeline,
            "silence_steps": int(silence_duration / dt),
        },
        "target_comp_id": BUILT_MODELS["target_nodes"][model_name],
        "net": BUILT_MODELS["mc_models"][model_name],
    }


def build_train_dataset(cfg_datasets) -> dict:
    """学習用の単一データセットを構築する"""
    return build_dataset(
        **cfg_datasets["common"],
        **cfg_datasets["train"],
    )


def build_steady_dataset(cfg_datasets: dict, amplitude: float) -> dict:
    return build_dataset(
        **cfg_datasets["common"],
        **cfg_datasets["sweep"]["common"],
        current_seed=0,  # placeholder
        pipeline=[
            {
                "_target_": "neurosurrogate.utils.current_generators.generate_steady",
                "value": amplitude,
            }
        ],
    )


def build_random_dataset(cfg_datasets: dict, current_seed: int) -> dict:
    return build_dataset(
        **cfg_datasets["common"],
        **cfg_datasets["sweep"]["common"],
        current_seed=current_seed,
        pipeline=[
            {
                "_target_": "neurosurrogate.utils.current_generators.generate_rand_pulse",
            }
        ],
    )


DATASETS_BUILDER_FUNCTIONS = {
    "steady": build_steady_dataset,
    "random": build_random_dataset,
}
SWEEP_VALUE_RESOLVERS = {
    "steady": lambda cfg: np.arange(
        cfg["start"], cfg["stop"] + cfg["step"], cfg["step"]
    ).tolist(),
    "random": lambda cfg: cfg,
}


def build_sweep_datasets(cfg_datasets) -> dict:
    sweep_cfg = cfg_datasets["sweep"]
    sweep_type = sweep_cfg["type"]

    builder = DATASETS_BUILDER_FUNCTIONS[sweep_type]
    resolver = SWEEP_VALUE_RESOLVERS[sweep_type]
    values = resolver(sweep_cfg["catalog"][sweep_type])

    datasets = {f"{sweep_type}_{v}": builder(cfg_datasets, v) for v in values}
    logger.info(f"Built {len(datasets)} datasets (type={sweep_type})")
    return datasets
