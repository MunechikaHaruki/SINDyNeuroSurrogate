import copy
import logging

import numpy as np
from numba import njit

from .neuron_core import (
    COMPARTMENT_TEMPLATES,
    HH_Params_numba,
    calc_hh_channel,
    calc_passive_channel,
)
from .xarray_utils import (
    build_indices,
    set_coords,
    set_i_internal,
)

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


def get_surrogate_network(
    origi_net: dict,
    origi_comp: dict,
    surr_indice: int,  # サロゲート化するノードのインデックスのリスト
    surr_gate_init: list | np.ndarray,  # 外から渡される潜在変数の初期値
):
    # ネットワークのディープコピー（元の配線図を汚さない）
    surr_net = copy.deepcopy(origi_net)
    origi_node_type = surr_net["nodes"][surr_indice]
    surr_net["nodes"][surr_indice] = "surr"

    # V の初期値を元のカタログから引き継ぐ
    origi_v_init = origi_comp[origi_node_type]["init"][0]

    # V の初期値と、外から来たゲート初期値を結合 (★カッコで囲んで安全に結合！)
    full_init = np.concatenate(
        (
            np.array([origi_v_init], dtype=np.float64),
            np.array(surr_gate_init, dtype=np.float64),
        )
    )

    # 潜在変数の次元数から、変数名(vars)とゲートフラグ(gates)を自動生成
    num_latents = len(surr_gate_init)
    surr_vars = ["V"] + [f"latent{i + 1}" for i in range(num_latents)]
    surr_gates = [False] + [True] * num_latents

    # 結合演算子 `|` を使って "surr" 部品を新規追加した新しいカタログを作る
    surr_comp = origi_comp | {
        "surr": {"init": full_init, "vars": surr_vars, "gate": surr_gates}
    }
    # 新しい配線図と、新しいカタログのセットを返す
    return surr_net, surr_comp


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


@njit
def dummy_theta(v, latent, i_int):
    return np.zeros(1, dtype=np.float64)


DUMMY_XI = np.zeros((2, 1), dtype=np.float64)

DUMMY_SINDY_ARGS = (DUMMY_XI, dummy_theta)


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


def unified_simulater(dt, u, net, surrogate_target=None, surrogate_model=None):
    params = HH_Params_numba()

    N = len(net["nodes"])
    C_matrix = calc_graph_laplacian(net["edges"], N)

    if surrogate_model is None:
        indice = build_indices(net, COMPARTMENT_TEMPLATES)
        net_args = (params, C_matrix, net["stim_node"])
        sindy_args = DUMMY_SINDY_ARGS
    else:
        surr_net, surr_comp = get_surrogate_network(
            net,
            COMPARTMENT_TEMPLATES,
            surrogate_target,
            surrogate_model.gate_init,
        )
        indice = build_indices(surr_net, surr_comp)
        net_args = (params, C_matrix, net["stim_node"])

        sindy_args = (
            surrogate_model.sindy.coefficients(),
            surrogate_model.compute_theta,
        )
    indice_args = (
        indice["gate_offsets"],
        indice["ids"]["passive"],
        indice["ids"]["hh"],
        indice["ids"]["surr"],
    )

    args = (net_args, indice_args, sindy_args)

    raw = generic_euler_solver(calc_universal_deriv, indice["init"], u, dt, args)

    dataset = set_coords(raw, u, indice["coords"], dt)
    dataset.attrs["dt"] = dt

    if surrogate_model is not None:
        dataset.attrs["surr_ids"] = indice["ids"]["surr"]

    I_ext_2d = np.zeros((len(u), N), dtype=np.float64)
    stim_idx = net["stim_node"]  # 設定から注入先を取得
    I_ext_2d[:, stim_idx] = u  # 指定されたコンパートメントにだけ u を流し込む
    set_i_internal(dataset, C_matrix, I_ext_2d)

    return dataset
