import logging
from collections import namedtuple

import numpy as np
from numba import njit

from .builder.build_coords import build_indices, set_coords, set_i_internal
from .model.model_compartments import (
    HH_Params_numba,
    calc_hh_channel,
    calc_passive_channel,
)
from .model.model_dataset import NeuronGraph
from .model.model_neurosindy import (
    DUMMY_SINDY_ARGS,
    DUMMY_SURR_COMP,
    SINDyNeuroSurrogate,
)

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
    [
        "C_matrix",
        "params",
        "stim_idx",
        "indice_args",
        "xi_matrix",
        "compute_theta",
        "gate_offsets",
    ],
)


@njit
def calc_universal_deriv(curr_x, u_t, model_args, dvar):
    """物理モデル用の汎用微分計算エンジン"""
    indice_args = model_args.indice_args
    N = model_args.C_matrix.shape[0]

    # 1. 電位ベクトルの抽出と網内電流の計算 (グラフラプラシアン)
    v_vec = curr_x[:N]
    I_internal = v_vec @ model_args.C_matrix
    I_internal[model_args.stim_idx] += u_t

    # 2. Passive コンパートメントの計算 (if分岐なしのSoA処理)
    for i in indice_args.passive:
        dvar[i] = calc_passive_channel(model_args.params, I_internal[i], v_vec[i])

    # 3. Hodgkin-Huxley コンパートメントの計算 (if分岐なしのSoA処理)
    for i in indice_args.hh:
        g_idx = model_args.gate_offsets[i]
        # HHは m, h, n の3つのゲート変数を持つ前提
        dvar[i] = calc_hh_channel(
            model_args.params,
            I_internal[i],
            v_vec[i],
            curr_x[g_idx : g_idx + 3],
            dvar[g_idx : g_idx + 3],
        )
    # SINDy代理モデルの計算
    for i in indice_args.surr:
        g_idx = model_args.gate_offsets[i]
        latent = curr_x[g_idx]  # サロゲートは1つの潜在変数 (latent) を持つ前提

        # 動的にコンパイルされた Theta 関数の呼び出し
        theta = model_args.compute_theta(v_vec[i], latent, I_internal[i])

        dvar[i] = model_args.xi_matrix[0] @ theta
        dvar[g_idx] = model_args.xi_matrix[1] @ theta


def unified_simulator(
    dt, u, net: NeuronGraph, surrogate_model: SINDyNeuroSurrogate = None
):
    if surrogate_model is None:
        sindy_args, surr_comp = DUMMY_SINDY_ARGS, DUMMY_SURR_COMP
    else:
        sindy_args, surr_comp = surrogate_model.sindy_args, surrogate_model.surr_comp
    params = HH_Params_numba()
    indice = build_indices(net, surr_comp)
    IndiceArgs = namedtuple("IndiceArgs", list(indice["ids"].keys()))
    raw = generic_euler_solver(
        indice["init"],
        u,
        dt,
        ModelArgs(
            params=params,
            C_matrix=net.graph_laplacian,
            stim_idx=net.stim_node_idx,
            indice_args=IndiceArgs(**indice["ids"]),
            xi_matrix=sindy_args[0],
            compute_theta=sindy_args[1],
            gate_offsets=indice["gate_offsets"],
        ),
    )
    print(f"surr_target_id:{indice['ids']['surr']}")
    dataset = set_coords(raw, u, indice["coords"], dt)
    set_i_internal(dataset, net.graph_laplacian, net.stim_node_idx, u)
    return dataset
