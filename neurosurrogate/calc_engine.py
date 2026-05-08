import logging
from collections import defaultdict, namedtuple

import numpy as np
from numba import njit

from .model import DUMMY_SINDY_ARGS, DUMMY_SURR_COMP, SINDyNeuroSurrogate
from .neuron_core import (
    COMPARTMENT_TEMPLATES,
    HH_Params_numba,
    calc_hh_channel,
    calc_passive_channel,
)
from .xarray_utils import StateAccumulator, set_coords, set_i_internal

logger = logging.getLogger(__name__)


@njit
def generic_euler_solver(init, u, dt, model_args):
    n_steps = len(u)
    n_vars = len(init)
    x_history = np.zeros((n_steps, n_vars))

    curr_x = init.copy()
    x_history[0] = curr_x
    dvar = np.zeros(n_vars)

    for t in range(n_steps - 1):
        # 微分計算関数の呼び出し。model_argsはタプル。
        calc_universal_deriv(curr_x, u[t], model_args, dvar)

        # 状態更新
        for i in range(n_vars):
            curr_x[i] += dvar[i] * dt
        x_history[t + 1] = curr_x

    return x_history


ModelArgs = namedtuple(
    "ModelArgs",
    ["C_matrix", "params", "stim_idx", "indice_args", "xi_matrix", "compute_theta"],
)

IndiceArgs = namedtuple(
    "IndiceArgs", ["gate_offsets", "passive_ids", "hh_ids", "surr_ids"]
)


@njit
def calc_universal_deriv(curr_x, u_t, model_args, dvar):
    """物理モデル用の汎用微分計算エンジン"""
    indice_args: IndiceArgs = model_args.indice_args
    N = model_args.C_matrix.shape[0]

    # 1. 電位ベクトルの抽出と網内電流の計算 (グラフラプラシアン)
    v_vec = curr_x[:N]
    I_internal = v_vec @ model_args.C_matrix
    I_internal[model_args.stim_idx] += u_t

    # 2. Passive コンパートメントの計算 (if分岐なしのSoA処理)
    for i in indice_args.passive_ids:
        dvar[i] = calc_passive_channel(model_args.params, I_internal[i], v_vec[i])

    # 3. Hodgkin-Huxley コンパートメントの計算 (if分岐なしのSoA処理)
    for i in indice_args.hh_ids:
        g_idx = indice_args.gate_offsets[i]
        # HHは m, h, n の3つのゲート変数を持つ前提
        dvar[i] = calc_hh_channel(
            model_args.params,
            I_internal[i],
            v_vec[i],
            curr_x[g_idx : g_idx + 3],
            dvar[g_idx : g_idx + 3],
        )
    # SINDy代理モデルの計算
    for i in indice_args.surr_ids:
        g_idx = indice_args.gate_offsets[i]
        latent = curr_x[g_idx]  # サロゲートは1つの潜在変数 (latent) を持つ前提

        # 動的にコンパイルされた Theta 関数の呼び出し
        theta = model_args.compute_theta(v_vec[i], latent, I_internal[i])

        dvar[i] = model_args.xi_matrix[0] @ theta
        dvar[g_idx] = model_args.xi_matrix[1] @ theta


def build_indices(nodes: list, surr_comp: dict):
    if surr_comp is None:
        compartments = COMPARTMENT_TEMPLATES
    else:
        compartments = COMPARTMENT_TEMPLATES | {"surr": surr_comp}

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
        "indice_args": IndiceArgs(
            gate_offsets=gate_offsets,
            passive_ids=ids["passive"],
            hh_ids=ids["hh"],
            surr_ids=ids["surr"],
        ),
        "init": acc.to_init(),
        "coords": acc.to_coords(),
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


def unified_simulator(dt, u, net, surrogate_model: SINDyNeuroSurrogate = None):
    if surrogate_model is None:
        sindy_args, surr_comp = DUMMY_SINDY_ARGS, DUMMY_SURR_COMP
    else:
        sindy_args, surr_comp = surrogate_model.sindy_args, surrogate_model.surr_comp

    params = HH_Params_numba()
    N = len(net["nodes"])
    C_matrix = calc_graph_laplacian(net["edges"], N)
    indice = build_indices(net["nodes"], surr_comp)
    raw = generic_euler_solver(
        indice["init"],
        u,
        dt,
        ModelArgs(
            params=params,
            C_matrix=C_matrix,
            stim_idx=net["stim_node"],
            indice_args=indice["indice_args"],
            xi_matrix=sindy_args[0],
            compute_theta=sindy_args[1],
        ),
    )
    dataset = set_coords(raw, u, indice["coords"], dt)
    set_i_internal(dataset, C_matrix, net["stim_node"], u)
    return dataset
