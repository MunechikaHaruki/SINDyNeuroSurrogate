from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr


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


def set_coords(raw, u, coords, dt):
    mindex_coords = xr.Coordinates.from_pandas_multiindex(coords, "features")

    # 2. Dataset 作成時に一括で定義する
    dataset = xr.Dataset(
        {
            "vars": (("time", "features"), raw),
            "I_ext": (("time"), u),
        },
        coords={
            "time": np.arange(len(u)) * dt,
            **mindex_coords,  # ここで一気にマルチインデックス化
        },
    )
    return dataset


def set_i_internal(dataset, C_matrix, I_ext_2d):
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
