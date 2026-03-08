import copy
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


COORDS = {
    "original": {
        "hh": {
            "variable": ["V", "M", "H", "N"],
            "comp_id": [0, 0, 0, 0],
            "gate": [False, True, True, True],
        },
        "hh3": {
            "variable": ["V_", "V", "V_", "M", "H", "N"],
            "comp_id": [0, 1, 2, 1, 1, 1],
            "gate": [False, False, False, True, True, True],
        },
    },
    "surrogate": {
        "hh": {
            "variable": ["V", "latent1"],
            "comp_id": [0, 0],
            "gate": [False, True],
        },
        "hh3": {
            "variable": ["V_", "V", "V_", "latent1"],
            "comp_id": [0, 1, 2, 1],
            "gate": [False, False, False, True],
        },
    },
}

COMPARTMENT_TEMPLATES = {
    "hh": {"init": np.array([v, m0(v_rel), h0(v_rel), n0(v_rel)])},
    "passive": {"init": np.array([E_REST])},
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

SURROGATE_TARGET = {"hh": [0], "hh3": [1]}


def get_surrogate_network(net: dict, surrogate_target: list):
    surrogate_net = copy.deepcopy(net)
    for target_idx in surrogate_target:
        # ★ ノードの種類を "surrogate" 部品に書き換える
        surrogate_net["nodes"][target_idx] = "surrogate"
    return surrogate_net


def build_indices(net: dict, compartments: dict):
    """
    ネットワーク配線図 (MC_MODELS) と 部品カタログ (COMPARTMENT_TEMPLATES) から
    Numba用のインデックス配列と初期値配列を自動生成する。
    """
    nodes = net["nodes"]
    N = len(nodes)

    gate_offsets = np.full(N, -1, dtype=np.int32)
    init_list = []

    # --- 修正1 & 2: 通常のリストとして初期化 ---
    ids_list = {}
    for k in compartments.keys():
        ids_list[k] = []

    # [Pass 1] 全ノードのVの初期値を配置し、IDを振り分ける
    for i, node_type in enumerate(nodes):
        # 電位の初期値を追加
        init_list.append(compartments[node_type]["init"][0])
        ids_list[node_type].append(i)  # 普通のリストなのでappend可能

    # [Pass 2] ゲート変数のオフセット計算と初期値の配置
    current_offset = N
    for i, node_type in enumerate(nodes):
        gate_inits = compartments[node_type]["init"][1:]

        # --- 修正3: ゲート変数が存在する場合のみオフセットを記録 ---
        if len(gate_inits) > 0:
            gate_offsets[i] = current_offset
            init_list.extend(gate_inits)
            current_offset += len(gate_inits)

    # 最後に、集めたIDリストを一気にNumPy配列(int32)に変換する
    ids = {k: np.array(v, dtype=np.int32) for k, v in ids_list.items()}

    return {
        "ids": ids,
        "gate_offsets": gate_offsets,
        "init": np.array(init_list, dtype=np.float64),
    }


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


ModeType = Literal["simulate", "surrogate"]


def unified_simulater(dt, u, data_type, mode: ModeType, **kwargs):
    net = MC_MODELS[data_type]
    params = HH_Params_numba()

    N = len(net["nodes"])
    C_matrix = calc_graph_laplacian(net["edges"], N)
    indice = build_indices(net, COMPARTMENT_TEMPLATES)
    if mode == "simulate":
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
        COORD = COORDS["original"][data_type]
    elif mode == "surrogate":
        args = (
            params,
            C_matrix,
            indice["ids"]["passive"],
            indice["ids"]["hh"],
            net["stim_node"],
            indice["gate_offsets"],
            kwargs["xi"],
            kwargs["compute_theta"],
        )
        gate_init = kwargs["gate_init"]
        if data_type == "hh3":
            init = np.concatenate(([-65, -65, -65], gate_init))
        elif data_type == "hh":
            init = np.concatenate(([-65], gate_init))
        deriv_func = calc_universal_surrogate
        COORD = COORDS["surrogate"][data_type]
    else:
        raise TypeError("Unsupported mode was detected")

    raw = generic_euler_solver(deriv_func, init, u, dt, args)

    mindex = pd.MultiIndex.from_arrays(
        [
            COORD["comp_id"],
            COORD["variable"],
            COORD["gate"],
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
    I_internal_np = V_data @ C_matrix

    # コンパートメントに対し、直接入力される電流をたす
    I_ext_2d = np.zeros((len(u), N), dtype=np.float64)
    stim_idx = net["stim_node"]  # 設定から注入先を取得
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
