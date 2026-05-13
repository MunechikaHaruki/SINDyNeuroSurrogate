import logging
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

from .builder.build_coords import build_indices, set_coords, set_i_internal
from .model.model_dataset import NeuronGraph
from .model.model_neurosindy import (
    DUMMY_SINDY_ARGS,
    SINDyNeuroSurrogate,
)
from .model.registry_compartments import (
    HHParams,
    calc_hh_channel,
    calc_passive_channel,
)

logger = logging.getLogger(__name__)


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


def calc_universal_deriv(curr_x, u_t, model_args):
    """物理モデル用の汎用微分計算エンジン"""
    indice_args = model_args.indice_args
    N = model_args.C_matrix.shape[0]

    # 1. 電位ベクトルの抽出と網内電流の計算 (グラフラプラシアン)
    v_vec = curr_x[:N]
    I_internal = v_vec @ model_args.C_matrix
    I_internal = I_internal.at[model_args.stim_idx].add(u_t)

    dvar = jnp.zeros_like(curr_x)

    # 2. Passive コンパートメントの計算
    for i in indice_args.passive:
        i = int(i)
        dvar = dvar.at[i].set(calc_passive_channel(model_args.params, I_internal[i], v_vec[i]))

    # 3. Hodgkin-Huxley コンパートメントの計算
    for i in indice_args.hh:
        i = int(i)
        g_idx = int(model_args.gate_offsets[i])
        dv, dgate = calc_hh_channel(
            model_args.params,
            I_internal[i],
            v_vec[i],
            curr_x[g_idx : g_idx + 3],
        )
        dvar = dvar.at[i].set(dv)
        dvar = dvar.at[g_idx : g_idx + 3].set(dgate)

    # 4. SINDy代理モデルの計算
    for i in indice_args.surr:
        i = int(i)
        g_idx = int(model_args.gate_offsets[i])
        latent = curr_x[g_idx]  # サロゲートは1つの潜在変数 (latent) を持つ前提

        theta = model_args.compute_theta(v_vec[i], latent, I_internal[i])

        dvar = dvar.at[i].set(model_args.xi_matrix[0] @ theta)
        dvar = dvar.at[g_idx].set(model_args.xi_matrix[1] @ theta)

    return dvar


def generic_euler_solver(init, u, dt, model_args):
    u_jax = jnp.array(u)
    init_jax = jnp.array(init)

    def step(curr_x, u_t):
        dvar = calc_universal_deriv(curr_x, u_t, model_args)
        new_x = curr_x + dvar * dt
        return new_x, curr_x

    # lax.scan でタイムループを実行: outputs[t] = curr_x before step t
    final_x, x_history_prefix = jax.lax.scan(step, init_jax, u_jax[:-1])
    x_history = jnp.concatenate([x_history_prefix, final_x[None]], axis=0)
    return np.array(x_history)


def unified_simulator(
    dt, u, net: NeuronGraph, surrogate_model: SINDyNeuroSurrogate = None
):
    sindy_args = surrogate_model.sindy_args if surrogate_model else DUMMY_SINDY_ARGS
    params = HHParams()

    indice = build_indices(net)
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
    logger.debug(f"surr_target_id:{indice['ids']['surr']}")
    dataset = set_coords(raw, u, indice["coords"], dt)
    set_i_internal(dataset, net.graph_laplacian, net.stim_node_idx, u)
    return dataset
