import logging
from typing import Literal

import numpy as np
from numba import float64, njit
from numba.experimental import jitclass

from .hh_utils import h0, m0, n0, tau_h, tau_m, tau_n
from .numba_core_utils import (
    build_indices,
    get_surrogate_network,
    set_coords,
    set_i_internal,
)

logger = logging.getLogger(__name__)


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
def calc_hh_channel(p, u_t, v, curr_gate, dvar_gate):
    m = curr_gate[0]
    h = curr_gate[1]
    n = curr_gate[2]
    v_rel = v - p.E_REST

    i_leak = p.G_LEAK * (v - p.E_LEAK)
    i_na = p.G_NA * m * m * m * h * (v - p.E_NA)
    i_k = p.G_K * n * n * n * n * (v - p.E_K)

    dv = (-i_leak - i_na - i_k + u_t) / p.C
    dvar_gate[0] = (1.0 / tau_m(v_rel)) * (-m + m0(v_rel))
    dvar_gate[1] = (1.0 / tau_h(v_rel)) * (-h + h0(v_rel))
    dvar_gate[2] = (1.0 / tau_n(v_rel)) * (-n + n0(v_rel))
    return dv


@njit
def calc_passive_channel(p, u_t, v):
    return (-p.G_LEAK * (v - p.E_LEAK) + u_t) / p.C


@njit
def calc_universal_simulate(curr_x, u_t, model_args, dvar):
    """物理モデル用の汎用微分計算エンジン"""
    p, C_matrix, passive_ids, hh_ids, stim_idx, gate_offsets = model_args
    N = C_matrix.shape[0]

    # 1. 電位ベクトルの抽出と網内電流の計算 (グラフラプラシアン)
    v_vec = curr_x[:N]
    I_internal = v_vec @ C_matrix
    I_internal[stim_idx] += u_t

    # 2. Passive コンパートメントの計算 (if分岐なしのSoA処理)
    for i in passive_ids:
        dvar[i] = calc_passive_channel(p, I_internal[i], v_vec[i])

    # 3. Hodgkin-Huxley コンパートメントの計算 (if分岐なしのSoA処理)
    for i in hh_ids:
        g_idx = gate_offsets[i]
        # HHは m, h, n の3つのゲート変数を持つ前提
        dvar[i] = calc_hh_channel(
            p,
            I_internal[i],
            v_vec[i],
            curr_x[g_idx : g_idx + 3],
            dvar[g_idx : g_idx + 3],
        )


@njit
def calc_universal_surrogate(curr_x, u_t, model_args, dvar):
    """SINDy代理モデル用の汎用微分計算エンジン"""
    (
        p,
        C_matrix,
        passive_ids,
        surr_ids,
        stim_idx,
        gate_offsets,
        xi_matrix,
        compute_theta,
    ) = model_args
    N = C_matrix.shape[0]

    # 1. 電位ベクトルの抽出と網内電流の計算 (グラフラプラシアン)
    v_vec = curr_x[:N]
    I_internal = v_vec @ C_matrix
    I_internal[stim_idx] += u_t

    # 2. Passive コンパートメントの計算
    for i in passive_ids:
        dvar[i] = calc_passive_channel(p, I_internal[i], v_vec[i])

    # 3. SINDy代理モデルの計算
    for i in surr_ids:
        g_idx = gate_offsets[i]
        latent = curr_x[g_idx]  # サロゲートは1つの潜在変数 (latent) を持つ前提

        # 動的にコンパイルされた Theta 関数の呼び出し
        theta = compute_theta(v_vec[i], latent, I_internal[i])

        dvar[i] = xi_matrix[0] @ theta
        dvar[g_idx] = xi_matrix[1] @ theta


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


E_REST = -65
v = -65
v_rel = v - E_REST


COMPARTMENT_TEMPLATES = {
    "hh": {
        "init": np.array([v, m0(v_rel), h0(v_rel), n0(v_rel)]),
        "vars": ["V", "M", "H", "N"],
        "gate": [False, True, True, True],
    },
    "passive": {"init": np.array([E_REST]), "vars": ["V"], "gate": [False]},
}

MC_MODELS = {
    "hh": {
        "nodes": ["hh"],
        "edges": [],
        "stim_node": 0,
    },
    "hh3": {
        "nodes": ["passive", "hh", "passive"],
        "edges": [(0, 1, 1.0), (1, 2, 0.7)],
        "stim_node": 0,
    },
}

SURROGATE_TARGET = {"hh": 0, "hh3": 1}


def calc_graph_laplacian(connections, N):
    G_matrix = np.zeros((N, N), dtype=np.float64)
    if N == 1 or connections is None:
        pass
    else:
        for i, j, g in connections:
            G_matrix[i, j] = G_matrix[j, i] = g
    D_matrix = np.diag(np.sum(G_matrix, axis=1))
    C_matrix = G_matrix - D_matrix  # 流入を正とするグラフラプラシアンの符号反転

    return C_matrix


def unified_simulater(
    dt, u, data_type, mode: Literal["simulate", "surrogate"], **kwargs
):
    net = MC_MODELS[data_type]
    params = HH_Params_numba()

    N = len(net["nodes"])
    C_matrix = calc_graph_laplacian(net["edges"], N)

    if mode == "simulate":
        indice = build_indices(net, COMPARTMENT_TEMPLATES)
        args = (
            params,
            C_matrix,
            indice["ids"]["passive"],
            indice["ids"]["hh"],
            net["stim_node"],
            indice["gate_offsets"],
        )
        init = indice["init"]
        deriv_func = calc_universal_simulate
    elif mode == "surrogate":
        surr_net, surr_comp = get_surrogate_network(
            net, COMPARTMENT_TEMPLATES, SURROGATE_TARGET[data_type], kwargs["gate_init"]
        )
        indice = build_indices(surr_net, surr_comp)
        args = (
            params,
            C_matrix,
            indice["ids"]["passive"],
            indice["ids"]["surr"],
            net["stim_node"],
            indice["gate_offsets"],
            kwargs["xi"],
            kwargs["compute_theta"],
        )
        init = indice["init"]
        deriv_func = calc_universal_surrogate
    else:
        raise TypeError("Unsupported mode was detected")

    raw = generic_euler_solver(deriv_func, init, u, dt, args)

    dataset = set_coords(raw, u, indice["coords"], dt)

    dataset.attrs = {
        "model_type": data_type,
        "mode": mode,
        "dt": dt,
    }

    # コンパートメント間を流れる電流の系間を流れる電流の計算
    v_dataset = dataset["vars"].sel(gate=False).sortby("comp_id")
    V_data = v_dataset.values  # 形状: (time, N)
    I_internal_np = V_data @ C_matrix

    # コンパートメントに対し、直接入力される電流をたす
    I_ext_2d = np.zeros((len(u), N), dtype=np.float64)
    stim_idx = net["stim_node"]  # 設定から注入先を取得
    I_ext_2d[:, stim_idx] = u  # 指定されたコンパートメントにだけ u を流し込む
    I_internal_np = I_internal_np + I_ext_2d

    set_i_internal(dataset, I_internal_np)

    return dataset
