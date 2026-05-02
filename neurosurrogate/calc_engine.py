import copy
import logging
from collections import defaultdict

import numpy as np
from numba import njit

from .model import DummySurrogate
from .neuron_core import (
    COMPARTMENT_TEMPLATES,
    HH_Params_numba,
    calc_hh_channel,
    calc_passive_channel,
    get_surr_comp,
)
from .xarray_utils import StateAccumulator, set_coords, set_i_internal

logger = logging.getLogger(__name__)


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


@njit
def calc_universal_deriv(curr_x, u_t, model_args, dvar):
    """物理モデル用の汎用微分計算エンジン"""
    net_args, indice_args, sindy_args = model_args
    gate_offsets, passive_ids, hh_ids, surr_ids = indice_args
    p, C_matrix, stim_idx = net_args
    xi_matrix, compute_theta = sindy_args
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
    # SINDy代理モデルの計算
    for i in surr_ids:
        g_idx = gate_offsets[i]
        latent = curr_x[g_idx]  # サロゲートは1つの潜在変数 (latent) を持つ前提

        # 動的にコンパイルされた Theta 関数の呼び出し
        theta = compute_theta(v_vec[i], latent, I_internal[i])

        dvar[i] = xi_matrix[0] @ theta
        dvar[g_idx] = xi_matrix[1] @ theta


def build_indices(nodes: list, surr_comp: dict):
    if surr_comp is None:
        compartments = COMPARTMENT_TEMPLATES
    else:
        compartments = COMPARTMENT_TEMPLATES | surr_comp

    N = len(nodes)
    gate_offsets = np.full(N, -1, dtype=np.int32)
    ids_list = {k: [] for k in compartments.keys()}
    acc = StateAccumulator()

    # [Pass 1] 電位変数の収集
    for i, node_type in enumerate(nodes):
        comp = compartments[node_type]
        acc.add(i, [comp["vars"][0]], [comp["gate"][0]], [comp["init"][0]])
        ids_list[node_type].append(i)

    # [Pass 2] ゲート変数の収集
    current_offset = N
    for i, node_type in enumerate(nodes):
        comp = compartments[node_type]
        gate_vars = comp["vars"][1:]
        if len(gate_vars) > 0:
            gate_offsets[i] = current_offset
            acc.add(i, comp["vars"][1:], comp["gate"][1:], comp["init"][1:])
            current_offset += len(gate_vars)

    ids = defaultdict(lambda: np.array([], dtype=np.int32))
    for k, v in ids_list.items():
        ids[k] = np.array(v, dtype=np.int32)

    return {
        "ids": ids,
        "gate_offsets": gate_offsets,
        "init": acc.to_init(),
        "coords": acc.to_coords(),
    }


def build_surrogate_net(origi_net, surr_indice):
    if surr_indice is None:
        return origi_net
    surr_net = copy.deepcopy(origi_net)
    surr_net["nodes"][surr_indice] = "surr"
    return surr_net


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


def unified_simulator(
    dt, u, net, surrogate_target=None, surrogate_model=DummySurrogate()
):
    params = HH_Params_numba()

    N = len(net["nodes"])
    C_matrix = calc_graph_laplacian(net["edges"], N)

    if surrogate_target is None:
        surr_comp = None
    else:
        surr_comp = get_surr_comp(
            net["nodes"][surrogate_target], surrogate_model.gate_init
        )

    surr_net = build_surrogate_net(net, surrogate_target)

    indice = build_indices(surr_net["nodes"], surr_comp)

    net_args = (params, C_matrix, net["stim_node"])
    indice_args = (
        indice["gate_offsets"],
        indice["ids"]["passive"],
        indice["ids"]["hh"],
        indice["ids"]["surr"],
    )
    args = (net_args, indice_args, surrogate_model.sindy_args)
    raw = generic_euler_solver(calc_universal_deriv, indice["init"], u, dt, args)
    dataset = set_coords(raw, u, indice["coords"], dt)

    I_ext_2d = np.zeros((len(u), N), dtype=np.float64)
    stim_idx = net["stim_node"]  # 設定から注入先を取得
    I_ext_2d[:, stim_idx] = u  # 指定されたコンパートメントにだけ u を流し込む
    set_i_internal(dataset, C_matrix, I_ext_2d)

    return dataset
