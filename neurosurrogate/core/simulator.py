import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .coords import GroupSpec, build_indices, set_coords, set_i_internal
from .network import DatasetConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelArgs:
    C_matrix: np.ndarray  # shape (N, N)       グラフラプラシアン
    stim_idx: int
    gate_offsets: np.ndarray  # shape (N,)      dtype=int32
    groups: dict[str, GroupSpec]  # type_name -> GroupSpec (kernel/idx/params/n_state)


def calc_universal_deriv(curr_x, u_t, ma):
    """全 GroupSpec に自身を apply させるだけ。type別分岐なし。"""
    N = ma.C_matrix.shape[0]
    v_vec = curr_x[:N]
    I_internal = (v_vec @ ma.C_matrix).at[ma.stim_idx].add(u_t)

    dvar = jnp.zeros_like(curr_x)
    for spec in ma.groups.values():
        dvar = spec.apply(dvar, curr_x, I_internal, v_vec, ma.gate_offsets)
    return dvar


def generic_euler_solver(init, u, dt, model_args):
    def step(curr_x, u_t):
        return curr_x + calc_universal_deriv(curr_x, u_t, model_args) * dt, curr_x

    # lax.scan でタイムループを実行: outputs[t] = curr_x before step t
    final_x, x_history_prefix = jax.lax.scan(step, jnp.array(init), jnp.array(u)[:-1])
    return np.array(jnp.concatenate([x_history_prefix, final_x[None]], axis=0))


def unified_simulator(cfg: DatasetConfig):
    """cfg.net の各 Compartment が kernel を保持している前提。surrogate も
    make_surr_comp で kernel 埋込済み Compartment を with_surrogates で挿入する"""
    net = cfg.net
    dt = cfg.dt
    u = cfg.build_current()
    indice = build_indices(net)
    dataset = set_coords(
        generic_euler_solver(
            indice["init"],
            u,
            dt,
            ModelArgs(
                C_matrix=net.graph_laplacian,
                stim_idx=net.stim_node_idx,
                gate_offsets=indice["gate_offsets"],
                groups=indice["groups"],
            ),
        ),
        u,
        indice["coords"],
        dt,
    )
    set_i_internal(dataset, net.graph_laplacian, net.stim_node_idx, u)
    return dataset
