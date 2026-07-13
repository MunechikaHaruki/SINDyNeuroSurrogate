import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .coords import collect_state_coords, set_coords, set_i_internal
from .network import Compartment, CompartmentType, DatasetConfig, NeuronGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupSpec:
    """同一 CompartmentType を共有する compartment 群を vmap 並列実行する 1 単位。
    型情報 (comp_type) と実行時データ (indices, params) を分けて保持。"""

    comp_type: CompartmentType  # 「型」= kernel + gate構造 + param_cls
    indices: np.ndarray  # shape (N_group,)
    params: (
        Any | None
    )  # batched NamedTuple (prefix (N_group,))、param_cls=None なら None

    def apply(
        self,
        dvar: jnp.ndarray,
        curr_x: jnp.ndarray,
        I_internal: jnp.ndarray,
        v_vec: jnp.ndarray,
        gate_offsets: np.ndarray,
    ) -> jnp.ndarray:
        """comp_type.kernel を vmap 展開して dvar を更新して返す"""
        idx = self.indices
        n_state = len(self.comp_type.gate_names)
        state_idx = gate_offsets[idx][:, None] + np.arange(n_state)
        in_axes = (None if self.params is None else 0, 0, 0, 0)
        dv, dstate = jax.vmap(self.comp_type.kernel, in_axes=in_axes)(
            self.params, I_internal[idx], v_vec[idx], curr_x[state_idx]
        )
        dvar = dvar.at[idx].set(dv)
        if n_state > 0:
            dvar = dvar.at[state_idx.ravel()].set(dstate.ravel())
        return dvar


def _group_by_type(
    nodes: list[Compartment],
) -> dict[str, list[tuple[int, Compartment]]]:
    """type_name → [(node_index, comp), ...] のバケット辞書。
    例: {"hh": [(1, hh_comp1), (3, hh_comp2)], "passive": [(0, p_comp)]}"""
    buckets: dict[str, list[tuple[int, Compartment]]] = {}
    for i, comp in enumerate(nodes):
        buckets.setdefault(comp.type.name, []).append((i, comp))
    return buckets


def _make_group_spec(bucket: list[tuple[int, Compartment]]) -> GroupSpec:
    """(index, comp) ペアのバケット → GroupSpec (batched params を作って kernel 準備)"""
    indices, comps = zip(*bucket, strict=True)
    comp_type = comps[0].type  # 同 type なので代表 comp から取得
    params = (
        None
        if comp_type.param_cls is None
        else jax.tree.map(
            lambda *xs: jnp.asarray(xs),
            *[
                c.params if c.params is not None else comp_type.default_params
                for c in comps
            ],
        )
    )
    return GroupSpec(
        comp_type=comp_type,
        indices=np.array(indices, dtype=np.int32),
        params=params,
    )


def build_model_state(net: NeuronGraph) -> dict:
    """NeuronGraph → シミュレータが必要とする全状態を構築。
    返却: {gate_offsets, init, coords, groups}"""
    acc, gate_offsets = collect_state_coords(net.nodes)
    buckets = _group_by_type(net.nodes)
    groups = {name: _make_group_spec(b) for name, b in buckets.items()}
    return {
        "gate_offsets": gate_offsets,
        "init": acc.to_init(),
        "coords": acc.to_coords(),
        "groups": groups,
    }


@dataclass(frozen=True)
class ModelArgs:
    C_matrix: np.ndarray  # shape (N, N)       グラフラプラシアン
    stim_idx: int
    gate_offsets: np.ndarray  # shape (N,)      dtype=int32
    groups: dict[str, GroupSpec]  # type_name -> GroupSpec
    stim_area_scale: float = 1.0  # u_ext を coupling と同スケールに揃える乗数


def calc_universal_deriv(curr_x, u_t, ma):
    """全 GroupSpec に自身を apply させるだけ。type別分岐なし。"""
    N = ma.C_matrix.shape[0]
    v_vec = curr_x[:N]
    I_internal = (v_vec @ ma.C_matrix).at[ma.stim_idx].add(u_t * ma.stim_area_scale)

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
    surr_comp_type で kernel 埋込済み CompartmentType を replace.apply で挿入する"""
    net = cfg.net
    dt = cfg.dt
    u = cfg.build_current()
    state = build_model_state(net)
    dataset = set_coords(
        generic_euler_solver(
            state["init"],
            u,
            dt,
            ModelArgs(
                C_matrix=net.graph_laplacian,
                stim_idx=net.stim_node_idx,
                gate_offsets=state["gate_offsets"],
                groups=state["groups"],
                stim_area_scale=net.stim_area_scale,
            ),
        ),
        u,
        state["coords"],
        dt,
    )
    set_i_internal(
        dataset, net.graph_laplacian, net.stim_node_idx, u, net.stim_area_scale
    )
    return dataset
