from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from numba import float64, njit, types
from numba.experimental import jitclass
from numba.typed import Dict
from omegaconf import OmegaConf

from .hh_utils import h0, m0, n0, tau_h, tau_m, tau_n

# jitclass for HH parameters
hh_params_spec = [
    ("E_REST", float64),
    ("C", float64),
    ("G_LEAK", float64),
    ("E_LEAK", float64),
    ("G_NA", float64),
    ("E_NA", float64),
    ("G_K", float64),
    ("E_K", float64),
    ("DT", float64),
]


@jitclass(hh_params_spec)
class HH_Params_numba:
    def __init__(self, params_dict):
        self.E_REST = params_dict.get("E_REST", -65.0)
        self.C = params_dict.get("C", 1.0)
        self.G_LEAK = params_dict.get("G_LEAK", 0.3)
        self.E_LEAK = params_dict.get("E_LEAK", 10.6 - 65.0)
        self.G_NA = params_dict.get("G_NA", 120.0)
        self.E_NA = params_dict.get("E_NA", 115.0 - 65.0)
        self.G_K = params_dict.get("G_K", 36.0)
        self.E_K = params_dict.get("E_K", -12.0 - 65.0)


# jitclass for Three-compartment model parameters
threecomp_params_spec = [
    ("hh", HH_Params_numba.class_type.instance_type),
    ("G_12", float64),
    ("G_23", float64),
]


@jitclass(threecomp_params_spec)
class ThreeComp_Params_numba:
    def __init__(self, params_dict):
        self.hh = HH_Params_numba(params_dict)
        self.G_12 = params_dict.get("G_12", 1)
        self.G_23 = params_dict.get("G_23", 0.7)


@njit
def initialize_hh(p):
    v = p.E_REST
    v_rel = v - p.E_REST
    return np.array([v, m0(v_rel), h0(v_rel), n0(v_rel)])


@njit
def initialize_hh3(p):
    init_hh = initialize_hh(p.hh)
    init_passiv_comp = np.array([p.hh.E_REST, p.hh.E_REST])
    return np.concatenate((init_hh, init_passiv_comp))


@njit
def calc_deriv_hh(curr_x, u_t, model_args, dvar):
    p = model_args[0]

    v = curr_x[0]
    m = curr_x[1]
    h = curr_x[2]
    n = curr_x[3]

    v_rel = v - p.E_REST

    i_leak = p.G_LEAK * (v - p.E_LEAK)
    i_na = p.G_NA * m * m * m * h * (v - p.E_NA)
    i_k = p.G_K * n * n * n * n * (v - p.E_K)

    dvar[0] = (-i_leak - i_na - i_k + u_t) / p.C
    dvar[1] = (1.0 / tau_m(v_rel)) * (-m + m0(v_rel))
    dvar[2] = (1.0 / tau_h(v_rel)) * (-h + h0(v_rel))
    dvar[3] = (1.0 / tau_n(v_rel)) * (-n + n0(v_rel))


@njit
def calc_deriv_hh3(curr_x, u_t, model_args, dvar):
    p = model_args[0]

    v_soma = curr_x[0]
    v_pre = curr_x[4]
    v_post = curr_x[5]

    i_pre = p.G_12 * (v_pre - v_soma)
    i_post = p.G_23 * (v_soma - v_post)

    soma_model_args = (p.hh,)
    calc_deriv_hh(curr_x, i_pre - i_post, soma_model_args, dvar[:4])

    dvar[4] = (-p.hh.G_LEAK * (v_pre - p.hh.E_LEAK) - i_pre + u_t) / p.hh.C
    dvar[5] = (-p.hh.G_LEAK * (v_post - p.hh.E_LEAK) + i_post) / p.hh.C


@njit
def calc_deriv_sindy(curr_x, u_t, model_args, dvar):
    params, xi_matrix, compute_theta = model_args
    theta = compute_theta(curr_x[0], curr_x[1], u_t)
    dvar[:] = xi_matrix @ theta


@njit
def calc_deriv_sindy_hh3(curr_x, u_t, model_args, dvar):
    params, xi_matrix, compute_theta = model_args

    v_soma = curr_x[0]
    latent = curr_x[1]
    v_pre = curr_x[2]
    v_post = curr_x[3]

    # 1. コンパートメント間の電流計算
    I_pre = params.G_12 * (v_pre - v_soma)
    I_post = params.G_23 * (v_soma - v_post)

    # 2. SINDyモデルによる微分値の計算 (dx = Xi @ Theta)
    theta = compute_theta(v_soma, latent, I_pre - I_post)

    dvar[0] = xi_matrix[0] @ theta
    dvar[1] = xi_matrix[1] @ theta
    dvar[2] = (
        -params.hh.G_LEAK * (v_pre - params.hh.E_LEAK) - I_pre + u_t
    ) / params.hh.C
    dvar[3] = (-params.hh.G_LEAK * (v_post - params.hh.E_LEAK) + I_post) / params.hh.C


@njit
def generic_euler_solver(deriv_func, init, u, dt, model_args):
    n_steps = len(u)
    n_vars = len(init)
    x_history = np.zeros((n_steps, n_vars))

    curr_x = init.copy()
    x_history[0] = curr_x
    dvar = np.zeros(n_vars)

    for t in range(n_steps - 1):
        # 微分計算関数の呼び出し。model_argsはタプル。
        deriv_func(curr_x, u[t], model_args, dvar)

        # 状態更新
        for i in range(n_vars):
            curr_x[i] += dvar[i] * dt
        x_history[t + 1] = curr_x

    return x_history


PARAMS_REGISTRY = {
    "hh": HH_Params_numba,
    "hh3": ThreeComp_Params_numba,
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


SIMULATER_CONFIGS = {
    "hh": {
        "deriv_func": calc_deriv_hh,
        "coords": {
            "variable": ["V_soma", "M", "H", "N"],
            "comp_part": ["soma", "soma", "soma", "soma"],
            "gate": [False, True, True, True],
        },
        "init_func": lambda p: initialize_hh(p),
    },
    "hh3": {
        "deriv_func": calc_deriv_hh3,
        "coords": {
            "variable": ["V_soma", "M", "H", "N", "V_pre", "V_post"],
            "comp_part": ["soma", "soma", "soma", "soma", "dend", "axon"],
            "gate": [False, True, True, True, False, False],
        },
        "init_func": lambda p: initialize_hh3(p),
    },
}

SURROGATER_CONFIGS = {
    "hh": {
        "deriv_func": calc_deriv_sindy,
        "coords": {
            "variable": ["V_soma", "latent1"],
            "comp_part": ["soma", "soma"],
            "gate": [False, True],
        },
    },
    "hh3": {
        "deriv_func": calc_deriv_sindy_hh3,
        "coords": {
            "variable": ["V_soma", "latent1", "V_pre", "V_post"],
            "comp_part": ["soma", "soma", "dend", "axon"],
            "gate": [False, True, False, False],
        },
    },
}


ModeType = Literal["simulate", "surrogate"]


def unified_simulater(dt, u, data_type, params_dict, mode: ModeType, **kwargs):
    params = instantiate_OmegaConf_params(params_dict, data_type=data_type)
    if mode == "simulate":
        CONF = SIMULATER_CONFIGS[data_type]
        args = (params,)
        init = CONF["init_func"](params)
    elif mode == "surrogate":
        CONF = SURROGATER_CONFIGS[data_type]
        args = (params, kwargs["xi"], kwargs["compute_theta"])
        init = kwargs["init"]
    else:
        raise TypeError("Unsupported mode was detected")

    raw = generic_euler_solver(CONF["deriv_func"], init, u, dt, args)

    mindex = pd.MultiIndex.from_arrays(
        [
            CONF["coords"]["comp_part"],
            CONF["coords"]["variable"],
            CONF["coords"]["gate"],
        ],
        names=("compartment", "variable", "gate"),
    )
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex, "features")

    # 2. Dataset 作成時に一括で定義する
    dataset = xr.Dataset(
        {
            "vars": (("time", "features"), raw),
            "I_ext": (("time"), u),
        },
        coords={
            "time": np.arange(len(u)) * dt,
            **mindex_coords,  # ここで一気にマルチインデックス化
        },
        attrs={
            "model_type": data_type,
            "mode": mode,
            "params": params_dict,
            "dt": dt,
        },
    )

    if data_type == "hh3":
        I_pre_np = params.G_12 * (
            dataset["vars"].sel(variable="V_pre", drop=True).squeeze().values
            - dataset["vars"].sel(variable="V_soma", drop=True).squeeze().values
        )
        I_post_np = params.G_23 * (
            dataset["vars"].sel(variable="V_soma", drop=True).squeeze().values
            - dataset["vars"].sel(variable="V_post", drop=True).squeeze().values
        )
        I_soma_np = I_pre_np - I_post_np

        internal_currents_data = np.stack([I_pre_np, I_post_np, I_soma_np], axis=0)

        dataset["I_internal"] = xr.DataArray(
            internal_currents_data,
            coords={
                "direction": ["pre", "post", "soma"],
                "time": dataset.time,
            },
            dims=["direction", "time"],
        )
    return dataset
