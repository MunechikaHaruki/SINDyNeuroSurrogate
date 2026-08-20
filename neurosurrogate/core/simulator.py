from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from .coords import collect_state_coords, set_coords, set_i_internal
from .network import Compartment, CompartmentType, DatasetConfig, NeuronGraph


@dataclass(frozen=True)
class _GroupSpec:
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


def _make_group_spec(bucket: list[tuple[int, Compartment]]) -> _GroupSpec:
    """(index, comp) ペアのバケット → _GroupSpec
    (batched params を作って kernel 準備)"""
    indices, comps = zip(*bucket, strict=True)
    comp_type = comps[0].type  # 同 type なので代表 comp から取得
    params = (
        None
        if comp_type.param_cls is None
        else jax.tree.map(
            lambda *xs: jnp.asarray(xs),
            *[
                c.params if c.params is not None else comp_type.param_cls()
                for c in comps
            ],
        )
    )
    return _GroupSpec(
        comp_type=comp_type,
        indices=np.array(indices, dtype=np.int32),
        params=params,
    )


def _graph_laplacian(net: NeuronGraph) -> np.ndarray:
    """形態 (edge の軸索 conductance [μS]) → `V @ L` が各ノードへの流入電流 [μA]
    になる対称ラプラシアン。**ネットを解く行列に畳むのはソルバの関心**なので、
    形態を持つ `NeuronGraph` でなくここに置く。"""
    N = len(net.nodes)
    G_matrix = np.zeros((N, N), dtype=np.float64)
    for e in net.edges:
        i, j = net.name_to_idx(e.src), net.name_to_idx(e.dst)
        G_matrix[i, j] = G_matrix[j, i] = e.weight
    return G_matrix - np.diag(
        np.sum(G_matrix, axis=1)
    )  # 流入を正とするグラフラプラシアンの符号反転


def _areas(net: NeuronGraph) -> np.ndarray:
    """ノード順の膜面積 [cm^2]。絶対量で来る coupling を kernel の規約
    (電流密度 [μA/cm^2]) へ直す除数。"""
    return np.array([c.area for c in net.nodes])


def _build_model_state(net: NeuronGraph) -> dict:
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
class _ModelArgs:
    C_matrix: np.ndarray  # shape (N, N)       グラフラプラシアン [μS]
    areas: np.ndarray  # shape (N,)          膜面積 [cm^2] (coupling の密度化)
    stim_idx: int
    gate_offsets: np.ndarray  # shape (N,)      dtype=int32
    groups: dict[str, _GroupSpec]  # type_name -> _GroupSpec


def _calc_universal_deriv(curr_x, u_t, ma):
    """全 _GroupSpec に自身を apply させるだけ。type別分岐なし。"""
    N = ma.C_matrix.shape[0]
    v_vec = curr_x[:N]
    # kernel は電流密度 [μA/cm^2] を受ける規約 → 絶対量で来る coupling をノードの
    # 面積で割ってから、既に密度の外部注入を足す (面積を持たない型は 1.0)。
    I_internal = (v_vec @ ma.C_matrix / ma.areas).at[ma.stim_idx].add(u_t)

    dvar = jnp.zeros_like(curr_x)
    for spec in ma.groups.values():
        dvar = spec.apply(dvar, curr_x, I_internal, v_vec, ma.gate_offsets)
    return dvar


def _generic_euler_solver(init, u, dt, model_args):
    def step(curr_x, u_t):
        return curr_x + _calc_universal_deriv(curr_x, u_t, model_args) * dt, curr_x

    # lax.scan でタイムループを実行: outputs[t] = curr_x before step t
    final_x, x_history_prefix = jax.lax.scan(step, jnp.array(init), jnp.array(u)[:-1])
    return np.array(jnp.concatenate([x_history_prefix, final_x[None]], axis=0))


def unified_simulator(cfg: DatasetConfig) -> xr.Dataset:
    """cfg.net の各 Compartment が kernel を保持している前提。surrogate も
    surr_comp_type で kernel 埋込済み CompartmentType を replace.apply で挿入する"""
    net = cfg.net
    dt = cfg.dt
    u = cfg.current
    state = _build_model_state(net)
    C_matrix = _graph_laplacian(net)
    node_areas = _areas(net)
    dataset = set_coords(
        _generic_euler_solver(
            state["init"],
            u,
            dt,
            _ModelArgs(
                C_matrix=C_matrix,
                areas=node_areas,
                stim_idx=cfg.stim_idx,
                gate_offsets=state["gate_offsets"],
                groups=state["groups"],
            ),
        ),
        u,
        state["coords"],
        dt,
    )
    set_i_internal(dataset, C_matrix, node_areas, cfg.stim_idx, u)
    return dataset
