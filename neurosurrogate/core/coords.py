from collections import defaultdict
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from ..registry.compartments import DEFAULT_PARAMS_BY_TYPE
from .network import NeuronGraph


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


def _stack_params_or_empty(items: list, type_name: str):
    """ノード別 params を stack。空なら shape (0,) の空 batch を返す (vmap size-0 用)"""
    default = DEFAULT_PARAMS_BY_TYPE[type_name]
    if items:
        return jax.tree.map(lambda *xs: jnp.asarray(xs), *items)
    return jax.tree.map(lambda x: jnp.empty((0,), dtype=jnp.asarray(x).dtype), default)


def build_indices(net: NeuronGraph):
    nodes = net.nodes
    N = len(nodes)
    gate_offsets = np.full(N, -1, dtype=np.int32)
    ids_list: dict[str, list] = {"hh": [], "passive": [], "surr": [], "traub": []}
    params_list: dict[str, list] = {"hh": [], "passive": [], "traub": []}
    acc = StateAccumulator()

    # [Pass 1] 電位変数の収集
    for i, comp in enumerate(nodes):
        acc.add(i, [comp.vars[0]], [comp.gate[0]], [comp.init[0]])
        ids_list[comp.type_name].append(i)
        if comp.type_name in params_list:
            params_list[comp.type_name].append(
                comp.params
                if comp.params is not None
                else DEFAULT_PARAMS_BY_TYPE[comp.type_name]
            )

    # [Pass 2] ゲート変数の収集
    current_offset = N
    for i, comp in enumerate(nodes):
        gate_vars = comp.vars[1:]
        if gate_vars:
            gate_offsets[i] = current_offset
            acc.add(i, comp.vars[1:], comp.gate[1:], comp.init[1:])
            current_offset += len(gate_vars)

    ids = defaultdict(lambda: np.array([], dtype=np.int32))
    for k, v in ids_list.items():
        ids[k] = np.array(v, dtype=np.int32)

    params_batched = {k: _stack_params_or_empty(v, k) for k, v in params_list.items()}

    return {
        "gate_offsets": gate_offsets,
        "ids": ids,
        "init": acc.to_init(),
        "coords": acc.to_coords(),
        "params": params_batched,
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
