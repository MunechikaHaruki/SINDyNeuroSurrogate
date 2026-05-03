import logging
from typing import Literal

import hydra
import numpy as np
import pysindy as ps
from conf import feature_library_components
from conf.feature_library_components import LIB_BUILDER_REGISTRY
from conf.neuron_models import MODEL_DEFINITIONS

from neurosurrogate.model import SINDyNeuroSurrogate

logger = logging.getLogger(__name__)


def build_surrogate(cfg_sindy):
    def _build_one(spec):
        builder = LIB_BUILDER_REGISTRY.get(spec["type"])
        if builder is None:
            raise ValueError(f"未知のlibrary type: {spec['type']}")
        return builder(spec)

    def _build_feature_library(library_specs):
        lib_input_pairs = [(_build_one(s), s["inputs"]) for s in library_specs]
        libraries, inputs = zip(*lib_input_pairs)
        return ps.GeneralizedLibrary(list(libraries), inputs_per_library=list(inputs))

    library = _build_feature_library(cfg_sindy["library_specs"])

    # preprocessorの初期化
    preprocessor = hydra.utils.instantiate(cfg_sindy["preprocessor"])
    # pySINDyの初期化
    initialized_sindy = ps.SINDy(
        feature_library=library,
        optimizer=hydra.utils.instantiate(cfg_sindy["optimizer"]),
    )

    # surrogate_modelの初期化
    return SINDyNeuroSurrogate(
        preprocessor, initialized_sindy, feature_library_components
    )


def build_simulator_config(dataset_cfg):
    def build_current_pipeline(current_cfg):
        iteration = current_cfg["iteration"]
        silence_steps = current_cfg["silence_steps"]
        dset_i_ext = np.zeros(iteration)

        for step_cfg in current_cfg["pipeline"]:
            func = hydra.utils.instantiate(step_cfg)
            func(dset_i_ext)

        dset_i_ext[:silence_steps] = 0
        dset_i_ext[-silence_steps:] = 0
        return dset_i_ext

    u = build_current_pipeline(dataset_cfg["current"])
    dt = dataset_cfg["dt"]
    parsed_dict = {"u": u, "dt": dt, "net": dataset_cfg["net"]}
    return parsed_dict


PIPE_FUNCS = {
    "steady": lambda amplitude: [
        {
            "_target_": "conf.current_generators.generate_steady",
            "value": amplitude,
        }
    ],
    "random": lambda seed: [
        {
            "_target_": "conf.current_generators.generate_rand_pulse",
            "seed": seed,
        }
    ],
}


def build_model(neuron_type):
    spec = MODEL_DEFINITIONS[neuron_type]
    nodes_dict = spec["nodes"]
    name_to_idx = {n: i for i, n in enumerate(nodes_dict.keys())}

    return {
        "name_to_idx_dict": name_to_idx,
        "nodes": list(nodes_dict.values()),
        "edges": [(name_to_idx[u], name_to_idx[v], g) for u, v, g in spec["edges"]],
        "stim_node": name_to_idx[spec["stim"]],
    }


CurrentType = Literal["steady", "random"]


def build_dataset(
    dt=0.01,
    silence_duration=80,
    duration=800,
    model_name="hh",
    pipeline=None,
    current_type=None,
    value=None,
) -> dict:
    """単一のケース設定(YAMLのcatalog_itemの階層構造そのまま)からデータセット辞書を構築する"""

    if pipeline is None:
        pipeline = PIPE_FUNCS[current_type](value)

    return {
        "data_type": model_name,
        "dt": dt,
        "current": {
            # フラットアクセスではなく、case_cfg["current"] のネストを参照する
            "iteration": int(duration / dt),
            "pipeline": pipeline,
            "silence_steps": int(silence_duration / dt),
        },
        "net": build_model(model_name),
    }
