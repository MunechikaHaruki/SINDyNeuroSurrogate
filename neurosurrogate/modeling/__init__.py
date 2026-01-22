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
    time_array = np.arange(len(i_ext)) * DT
    # Preprocess the simulation data
    return preprocess_dataset(
        data_type, i_ext, results, neuron_cfg, time_array, surrogate=False
    )


def predict(init, dt, iter, u, sindy, data_type=None, params_dict=None):
    if hasattr(init, "to_numpy"):
        init = init.to_numpy()
    if hasattr(u, "to_numpy"):
        u = u.to_numpy()
    # ensure they are numpy arrays
    init = np.asarray(init)
    if u is not None:
        u = np.asarray(u)

    if data_type == "hh3":
        logger.info("hh3のサロゲートモデルをテストします")
        init = np.array([init[0], init[1], -65, -65])  # v,隠れ変数,v_pre,v_post
        params = instantiate_OmegaConf_params(params_dict, data_type=data_type)
        var = simulate_three_comp_numba(
            init=init,
            u=u,
            xi_matrix=sindy.coefficients(),
            params=params,
            dt=dt,
        )
    elif data_type == "hh":
        logger.info("hhのサロゲートモデルをテストします")
        var = simulate_sindy(init, u, sindy.coefficients(), dt)
    else:
        raise ValueError(f"未知のmodeが指定されました: {data_type}")
    time = np.arange(0, iter * dt, dt)
    return preprocess_dataset(
        model_type=data_type,
        i_ext=u,
        results=var,
        params=params_dict,
        time_array=time,
        surrogate=True,
    )
