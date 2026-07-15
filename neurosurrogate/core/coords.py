from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from . import access
from .network import Compartment


@dataclass
class StateAccumulator:
    """変数座標 (comp_id/variable/gate/init) を蓄積する helper。"""

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


def collect_state_coords(
    nodes: list[Compartment],
) -> tuple[StateAccumulator, np.ndarray]:
    """変数座標 (comp_id/variable/gate/init) と gate_offsets を収集。
    レイアウト: [0..N-1] 電位ブロック → [N..] 各 comp のゲート/状態変数を順次配置"""
    N = len(nodes)
    gate_offsets = np.full(N, -1, dtype=np.int32)
    acc = StateAccumulator()

    # [Pass 1] 電位変数
    for i, comp in enumerate(nodes):
        t = comp.type
        acc.add(i, [t.vars[0]], [t.gate[0]], [t.init[0]])

    # [Pass 2] ゲート/状態変数
    current_offset = N
    for i, comp in enumerate(nodes):
        t = comp.type
        if len(t.vars) > 1:
            gate_offsets[i] = current_offset
            acc.add(i, t.vars[1:], t.gate[1:], t.init[1:])
            current_offset += len(t.vars) - 1

    return acc, gate_offsets


def set_coords(raw, u, coords, dt) -> xr.Dataset:
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


def set_latent_coords(
    v: np.ndarray, latent: np.ndarray, u: np.ndarray, comp_id: int, dt: float
) -> xr.Dataset:
    """単一 comp の [V, latent1..N] を preprocessed Dataset に組立 (surrogate用)。"""
    n_latent = latent.shape[1]
    acc = StateAccumulator()
    acc.add(comp_id, ["V"], [False], [0.0])
    acc.add(
        comp_id,
        [f"latent{i + 1}" for i in range(n_latent)],
        [True] * n_latent,
        [0.0] * n_latent,
    )
    return set_coords(
        raw=np.concatenate((v.reshape(-1, 1), latent), axis=1),
        u=u,
        coords=acc.to_coords(),
        dt=dt,
    )


def set_i_internal(dataset, C_matrix, stim_idx, u, stim_area_scale: float = 1.0):
    N = C_matrix.shape[0]
    I_ext_2d = np.zeros((len(u), N), dtype=np.float64)
    I_ext_2d[:, stim_idx] = u * stim_area_scale  # 密度→絶対変換 (traub19 等)

    V_data = access.potential_matrix(dataset)  # 形状: (time, N)
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
