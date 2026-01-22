import numpy as np
from loguru import logger
from numba import types
from numba.typed import Dict
from omegaconf import OmegaConf

from ..utils.data_processing import preprocess_dataset
from ._simulater import (
    HH_Params_numba,
    ThreeComp_Params_numba,
    hh3_simulate_numba,
    hh_simulate_numba,
)
from ._surrogate import simulate_sindy, simulate_three_comp_numba

PARAMS_REGISTRY = {
    "hh": HH_Params_numba,
    "hh3": ThreeComp_Params_numba,
}
SIMULATOR_REGISTRY = {
    "hh": hh_simulate_numba,
    "hh3": hh3_simulate_numba,
}
SURROGATER_REGISTRY = {"hh": simulate_sindy, "hh3": simulate_three_comp_numba}


def instantiate_OmegaConf_params(cfg, data_type):
    if cfg is None:
        py_dict = {}
    else:
        py_dict = OmegaConf.to_container(cfg, resolve=True)
    nb_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    for k, v in py_dict.items():
        nb_dict[k] = float(v)
    return PARAMS_REGISTRY[data_type](nb_dict)


def simulater(
    neuron_cfg,
    data_type,
    DT,
    i_ext,
):
    params = instantiate_OmegaConf_params(neuron_cfg, data_type=data_type)
    results = SIMULATOR_REGISTRY[data_type](i_ext, params, DT)
    # Preprocess the simulation data
    return preprocess_dataset(
        model_type=data_type,
        i_ext=i_ext,
        results=results,
        params=neuron_cfg,
        dt=DT,
        surrogate=False,
    )


def predict(init, dt, u, sindy, params_dict, data_type):
    logger.info(f"{data_type}のサロゲートモデルをテスト")
    params = instantiate_OmegaConf_params(params_dict, data_type=data_type)
    var = SURROGATER_REGISTRY[data_type](
        init=init,
        u=u,
        xi_matrix=sindy.coefficients(),
        params=params,
        dt=dt,
    )

    return preprocess_dataset(
        model_type=data_type,
        i_ext=u,
        results=var,
        params=params_dict,
        dt=dt,
        surrogate=True,
    )
