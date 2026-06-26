import logging
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .builder.build_coords import build_indices, set_coords, set_i_internal
from .model.model_dataset import DatasetConfig
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


@dataclass(frozen=True)
class IndiceArgs:
    hh: np.ndarray  # shape (N_hh,)  dtype=int32
    passive: np.ndarray  # shape (N_p,)   dtype=int32
    surr: np.ndarray  # shape (N_s,)   dtype=int32


@dataclass(frozen=True)
class ModelArgs:
    C_matrix: np.ndarray  # shape (N, N)       グラフラプラシアン
    params: HHParams
    stim_idx: int
    indice_args: IndiceArgs
    xi_matrix: np.ndarray  # shape (2, n_terms)  SINDy係数行列
    compute_theta: Callable  # (v, latent, i_t) -> jnp.ndarray
    gate_offsets: np.ndarray  # shape (N,)          dtype=int32


def calc_universal_deriv(curr_x, u_t, model_args):
    """物理モデル用の汎用微分計算エンジン"""
    indice_args = model_args.indice_args
    N = model_args.C_matrix.shape[0]

    # 1. 電位ベクトルの抽出と網内電流の計算 (グラフラプラシアン)
    v_vec = curr_x[:N]
    I_internal = v_vec @ model_args.C_matrix
    I_internal = I_internal.at[model_args.stim_idx].add(u_t)

    dvar = jnp.zeros_like(curr_x)

    # 2. Passive: vmap で一括処理
    p_idx = indice_args.passive
    dvar = dvar.at[p_idx].set(
        jax.vmap(lambda i_t, v: calc_passive_channel(model_args.params, i_t, v))(
            I_internal[p_idx], v_vec[p_idx]
        )
    )

    # 3. HH: ゲートインデックス行列 (N_hh, 3) を静的に計算して vmap
    hh_idx = indice_args.hh
    gate_idx = model_args.gate_offsets[hh_idx][:, None] + np.arange(3)  # (N_hh, 3)
    dv_hh, dgate_hh = jax.vmap(
        lambda i_t, v, g: calc_hh_channel(model_args.params, i_t, v, g)
    )(I_internal[hh_idx], v_vec[hh_idx], curr_x[gate_idx])
    dvar = dvar.at[hh_idx].set(dv_hh)
    dvar = dvar.at[gate_idx.ravel()].set(dgate_hh.ravel())

    # 4. SINDy: 潜在変数を一括抽出して vmap
    surr_idx = indice_args.surr
    g_off_s = model_args.gate_offsets[surr_idx]

    def surr_one(i_t, v, lat):
        theta = model_args.compute_theta(v, lat, i_t)
        return model_args.xi_matrix[0] @ theta, model_args.xi_matrix[1] @ theta

    dv_s, dlat_s = jax.vmap(surr_one)(
        I_internal[surr_idx], v_vec[surr_idx], curr_x[g_off_s]
    )
    dvar = dvar.at[surr_idx].set(dv_s)
    dvar = dvar.at[g_off_s].set(dlat_s)

    return dvar


def generic_euler_solver(init, u, dt, model_args):
    def step(curr_x, u_t):
        return curr_x + calc_universal_deriv(curr_x, u_t, model_args) * dt, curr_x

    # lax.scan でタイムループを実行: outputs[t] = curr_x before step t
    final_x, x_history_prefix = jax.lax.scan(step, jnp.array(init), jnp.array(u)[:-1])
    return np.array(jnp.concatenate([x_history_prefix, final_x[None]], axis=0))


def unified_simulator(
    cfg: DatasetConfig, surrogate_model: SINDyNeuroSurrogate | None = None
):
    sindy_args = surrogate_model.sindy_args if surrogate_model else DUMMY_SINDY_ARGS
    net = cfg.net
    dt = cfg.dt
    u = cfg.current.build(dt)
    indice = build_indices(net)
    logger.debug(f"surr_target_id:{indice['ids']['surr']}")
    dataset = set_coords(
        generic_euler_solver(
            indice["init"],
            u,
            dt,
            ModelArgs(
                params=HHParams(),
                C_matrix=net.graph_laplacian,
                stim_idx=net.stim_node_idx,
                indice_args=IndiceArgs(**indice["ids"]),
                xi_matrix=sindy_args[0],
                compute_theta=sindy_args[1],
                gate_offsets=indice["gate_offsets"],
            ),
        ),
        u,
        indice["coords"],
        dt,
    )
    set_i_internal(dataset, net.graph_laplacian, net.stim_node_idx, u)
    return dataset
