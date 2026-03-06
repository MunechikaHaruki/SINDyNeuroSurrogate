from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from numba import float64, njit
from numba.experimental import jitclass

from .hh_utils import h0, m0, n0, tau_h, tau_m, tau_n


@jitclass(
    [
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
)
class HH_Params_numba:
    def __init__(self):
        self.E_REST = -65.0
        self.C = 1.0
        self.G_LEAK = 0.3
        self.E_LEAK = 10.6 - 65.0
        self.G_NA = 120.0
        self.E_NA = 115.0 - 65.0
        self.G_K = 36.0
        self.E_K = -12.0 - 65.0


@njit
def initialize_hh(p):
    v = p.E_REST
    v_rel = v - p.E_REST
    return np.array([v, m0(v_rel), h0(v_rel), n0(v_rel)])


@njit
def initialize_hh3(p):
    v = p.E_REST
    v_rel = v - p.E_REST
    return np.array([p.E_REST, v, p.E_REST, m0(v_rel), h0(v_rel), n0(v_rel)])


@njit
def calc_deriv_hh(curr_x, u_t, model_args, dvar):
    p, G_matrix, _, _ = model_args

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
def calc_hh_channel(p, u_t, v, m, h, n):
    v_rel = v - p.E_REST

    i_leak = p.G_LEAK * (v - p.E_LEAK)
    i_na = p.G_NA * m * m * m * h * (v - p.E_NA)
    i_k = p.G_K * n * n * n * n * (v - p.E_K)

    dv = (-i_leak - i_na - i_k + u_t) / p.C
    dm = (1.0 / tau_m(v_rel)) * (-m + m0(v_rel))
    dh = (1.0 / tau_h(v_rel)) * (-h + h0(v_rel))
    dn = (1.0 / tau_n(v_rel)) * (-n + n0(v_rel))

    return dv, dm, dh, dn


@njit
def calc_deriv_hh3(curr_x, u_t, model_args, dvar):
    p, G_matrix, _, _ = model_args

    v_soma = curr_x[1]
    v_pre = curr_x[0]
    v_post = curr_x[2]

    I_pre = G_matrix[0][1] * (v_pre - v_soma)
    I_post = G_matrix[1][2] * (v_soma - v_post)

    dv, dm, dh, dn = calc_hh_channel(
        p, I_pre - I_post, v_soma, curr_x[3], curr_x[4], curr_x[5]
    )

    dvar[0] = (-p.G_LEAK * (v_pre - p.E_LEAK) - I_pre + u_t) / p.C
    dvar[1] = dv
    dvar[2] = (-p.G_LEAK * (v_post - p.E_LEAK) + I_post) / p.C
    dvar[3] = dm
    dvar[4] = dh
    dvar[5] = dn


@njit
def calc_deriv_sindy(curr_x, u_t, model_args, dvar):
    params, G_matrix, xi_matrix, compute_theta = model_args
    theta = compute_theta(curr_x[0], curr_x[1], u_t)
    dvar[:] = xi_matrix @ theta


@njit
def calc_deriv_sindy_hh3(curr_x, u_t, model_args, dvar):
    params, G_matrix, xi_matrix, compute_theta = model_args

    v_soma = curr_x[1]
    latent = curr_x[3]
    v_pre = curr_x[0]
    v_post = curr_x[2]

    # 1. コンパートメント間の電流計算
    I_pre = G_matrix[0][1] * (v_pre - v_soma)
    I_post = G_matrix[1][2] * (v_soma - v_post)

    # 2. SINDyモデルによる微分値の計算 (dx = Xi @ Theta)
    theta = compute_theta(v_soma, latent, I_pre - I_post)

    dvar[1] = xi_matrix[0] @ theta
    dvar[3] = xi_matrix[1] @ theta
    dvar[0] = (-params.G_LEAK * (v_pre - params.E_LEAK) - I_pre + u_t) / params.C
    dvar[2] = (-params.G_LEAK * (v_post - params.E_LEAK) + I_post) / params.C


@njit
def generic_euler_solver(deriv_func, init, u, dt, model_args):
    n_steps = len(u)
    n_vars = len(init)
    x_history = np.zeros((n_steps, n_vars))

    curr_x = init.copy()
    x_history[0] = curr_x
    dvar = np.zeros(n_vars)

    for t in range(n_steps - 1):
        if t < 3:
            print("Step:", t)
            print("curr_x:", curr_x)
            print("dvar:", dvar)
        # 微分計算関数の呼び出し。model_argsはタプル。
        deriv_func(curr_x, u[t], model_args, dvar)

        # 状態更新
        for i in range(n_vars):
            curr_x[i] += dvar[i] * dt
        x_history[t + 1] = curr_x

    return x_history


SIMULATER_CONFIGS = {
    "hh": {
        "deriv_func": calc_deriv_hh,
        "coords": {
            "variable": ["V", "M", "H", "N"],
            "comp_id": [0, 0, 0, 0],
            "gate": [False, True, True, True],
        },
        "init_func": lambda p: initialize_hh(p),
        "N": 1,
        "connections": None,
        "stim_comp_id": 0,
        "surrogate": {
            "deriv_func": calc_deriv_sindy,
            "coords": {
                "variable": ["V", "latent1"],
                "comp_id": [0, 0],
                "gate": [False, True],
            },
        },
    },
    "hh3": {
        "deriv_func": calc_deriv_hh3,
        "coords": {
            "variable": ["V_", "V", "V_", "M", "H", "N"],
            "comp_id": [0, 1, 2, 1, 1, 1],
            "gate": [False, False, False, True, True, True],
        },
        "init_func": lambda p: initialize_hh3(p),
        "N": 3,
        "connections": [
            (
                0,
                1,
                1,
            ),  # 接続情報のリスト（エッジリスト） 書式：(接続元インデックス, 接続先インデックス, コンダクタンス)
            (1, 2, 0.7),
        ],
        "stim_comp_id": 0,
        "surrogate": {
            "deriv_func": calc_deriv_sindy_hh3,
            "coords": {
                "variable": ["V_", "V", "V_", "latent1"],
                "comp_id": [0, 1, 2, 1],
                "gate": [False, False, False, True],
            },
        },
    },
}


def generate_G_matrix(connections, N):
    G_matrix = np.zeros((N, N), dtype=np.float64)
    if N == 1 or connections is None:
        return G_matrix
    for i, j, g in connections:
        G_matrix[i, j] = G_matrix[j, i] = g
    return G_matrix


ModeType = Literal["simulate", "surrogate"]


def unified_simulater(dt, u, data_type, mode: ModeType, **kwargs):
    CONF = SIMULATER_CONFIGS[data_type]
    params = HH_Params_numba()

    connections = CONF["connections"]
    N = CONF["N"]
    G_matrix = generate_G_matrix(connections, N)

    if mode == "simulate":
        args = (params, G_matrix, None, None)
        init = CONF["init_func"](params)
        deriv_func = CONF["deriv_func"]
        COORDS = CONF["coords"]
    elif mode == "surrogate":
        args = (params, G_matrix, kwargs["xi"], kwargs["compute_theta"])
        init = kwargs["init"]
        deriv_func = CONF["surrogate"]["deriv_func"]
        COORDS = CONF["surrogate"]["coords"]
    else:
        raise TypeError("Unsupported mode was detected")

    raw = generic_euler_solver(deriv_func, init, u, dt, args)

    mindex = pd.MultiIndex.from_arrays(
        [
            COORDS["comp_id"],
            COORDS["variable"],
            COORDS["gate"],
        ],
        names=("comp_id", "variable", "gate"),
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
            "dt": dt,
        },
    )

    # コンパートメント間を流れる電流の系間を流れる電流の計算
    v_dataset = dataset["vars"].sel(gate=False).sortby("comp_id")
    V_data = v_dataset.values  # 形状: (time, N)
    I_internal_np = (V_data @ G_matrix) - (V_data * np.sum(G_matrix, axis=1))

    # コンパートメントに対し、直接入力される電流をたす
    I_ext_2d = np.zeros((len(u), N), dtype=np.float64)
    stim_idx = CONF.get("stim_comp_id", 0)  # 設定から注入先を取得
    I_ext_2d[:, stim_idx] = u  # 指定されたコンパートメントにだけ u を流し込む
    I_internal_np = I_internal_np + I_ext_2d
    # xarray に格納
    dataset["I_internal"] = xr.DataArray(
        I_internal_np.T,  # (N, time) の形状にするため転置
        coords={
            "node_id": np.arange(N),
            "time": dataset.time,
        },
        dims=["node_id", "time"],
    )
    return dataset
