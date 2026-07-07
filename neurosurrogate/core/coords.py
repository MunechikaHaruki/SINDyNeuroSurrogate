from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from .network import CompartmentType, NeuronGraph


@dataclass(frozen=True)
class GroupSpec:
    """同一 CompartmentType を共有する compartment 群を vmap 並列実行する 1 単位。
    型情報 (comp_type) と実行時データ (indices, params) を分けて保持。"""

    comp_type: CompartmentType  # 「型」= kernel + gate構造 + param_cls
    indices: np.ndarray  # shape (N_group,)
    params: Any | None  # batched NamedTuple (prefix (N_group,))、param_cls=None なら None

    @classmethod
    def from_comps(cls, comps: list, indices: np.ndarray) -> "GroupSpec":
        """同type Compartment 群 → 1つの GroupSpec に集約。
        全 comp が同じ CompartmentType を共有する前提。"""
        comp_type = comps[0].type
        if comp_type.param_cls is None:
            params = None
        else:
            params = jax.tree.map(
                lambda *xs: jnp.asarray(xs), *[c.resolved_params for c in comps]
            )
        return cls(comp_type=comp_type, indices=indices, params=params)

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


@dataclass
class StateAccumulator:
    comp_id: list = field(default_factory=list)
    variable: list = field(default_factory=list)
    gate: list = field(default_factory=list)
    init: list = field(default_factory=list)

    def add(self, comp_id, variables, gates, inits):
        self.comp_id.extend([comp_id] * len(variables))
        self.variable.extend(variables)
        self.gate.extend(gates)
        self.init.extend(inits)

    def to_coords(self):
        return pd.MultiIndex.from_arrays(
            [self.comp_id, self.variable, self.gate],
            names=("comp_id", "variable", "gate"),
        )

    def to_init(self):
        return np.array(self.init, dtype=np.float64)


def build_indices(net: NeuronGraph):
    nodes = net.nodes
    N = len(nodes)
    gate_offsets = np.full(N, -1, dtype=np.int32)
    # type_name で動的にグルーピング (ハードコード脱却)
    grouped: dict[str, list[tuple[int, Any]]] = {}
    acc = StateAccumulator()

    # [Pass 1] 電位変数の収集 + type別グルーピング
    for i, comp in enumerate(nodes):
        acc.add(i, [comp.vars[0]], [comp.gate[0]], [comp.init[0]])
        grouped.setdefault(comp.type_name, []).append((i, comp))

    # [Pass 2] ゲート/状態変数の収集
    current_offset = N
    for i, comp in enumerate(nodes):
        gate_vars = comp.vars[1:]
        if gate_vars:
            gate_offsets[i] = current_offset
            acc.add(i, comp.vars[1:], comp.gate[1:], comp.init[1:])
            current_offset += len(gate_vars)

    groups: dict[str, GroupSpec] = {
        type_name: GroupSpec.from_comps(
            comps=[c for _, c in items],
            indices=np.array([i for i, _ in items], dtype=np.int32),
        )
        for type_name, items in grouped.items()
    }

    return {
        "gate_offsets": gate_offsets,
        "groups": groups,
        "init": acc.to_init(),
        "coords": acc.to_coords(),
    }


def set_coords(raw, u, coords, dt):
    return xr.Dataset(
        {
            "vars": (("time", "features"), raw),
            "I_ext": (("time"), u),
        },
        coords={
            "time": np.arange(len(u)) * dt,
            **xr.Coordinates.from_pandas_multiindex(
                coords, "features"
            ),  # ここで一気にマルチインデックス化
        },
    )


def set_i_internal(dataset, C_matrix, stim_idx, u):
    N = C_matrix.shape[0]
    I_ext_2d = np.zeros((len(u), N), dtype=np.float64)
    I_ext_2d[:, stim_idx] = u  # 指定されたコンパートメントにだけ u を流し込む

    V_data = dataset["vars"].sel(gate=False).sortby("comp_id").values  # 形状: (time, N)
    I_internal_np = V_data @ C_matrix + I_ext_2d
    # xarray に格納
    dataset["I_internal"] = xr.DataArray(
        I_internal_np,  # (N, time) の形状にするため転置
        coords={
            "time": dataset.time,
            "node_id": np.arange(I_internal_np.shape[1]),
        },
        dims=["time", "node_id"],
    )
