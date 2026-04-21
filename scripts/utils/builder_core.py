import logging
import random

import hydra
import numpy as np
import pysindy as ps
from conf.feature_library_components import LIB_BUILDER_REGISTRY

from neurosurrogate.modeling import SINDySurrogateWrapper
from neurosurrogate.modeling.neuron_core import FUNC_COST_MAP, HH_COST

logger = logging.getLogger(__name__)


def build_simulator_config(dataset_cfg):
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

    u = build_current_pipeline(dataset_cfg["current"])
    dt = dataset_cfg["dt"]
    parsed_dict = {"u": u, "dt": dt, "net": dataset_cfg["net"]}
    return parsed_dict


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

    # pySINDyの初期化
    initialized_sindy = ps.SINDy(
        feature_library=library,
        optimizer=hydra.utils.instantiate(cfg_sindy["optimizer"]),
    )
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.conf import feature_library_components

    # surrogate_modelの初期化
    return SINDySurrogateWrapper(
        initialized_sindy, feature_library_components, FUNC_COST_MAP, HH_COST
    )
