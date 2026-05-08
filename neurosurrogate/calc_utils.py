from collections import defaultdict
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class OpCost:
    exp: int = 0
    div: int = 0
    pm: int = 0
    mul: int = 0

    def __add__(self, other: "OpCost") -> "OpCost":
        return OpCost(
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in fields(self)
            }
        )

    def __mul__(self, n: int) -> "OpCost":
        return OpCost(**{f.name: getattr(self, f.name) * n for f in fields(self)})

    def to_dict(self) -> dict[str, int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


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


def build_indices(nodes: list, surr_comp: dict):
    from .neuron_core import COMPARTMENT_TEMPLATES

    if surr_comp is None:
        compartments = COMPARTMENT_TEMPLATES
    else:
        compartments = COMPARTMENT_TEMPLATES | {"surr": surr_comp}

    N = len(nodes)
    gate_offsets = np.full(N, -1, dtype=np.int32)
    ids_list = {k: [] for k in compartments.keys()}
    acc = StateAccumulator()

    # [Pass 1] 電位変数の収集
    for i, node_type in enumerate(nodes):
        comp = compartments[node_type]
        acc.add(i, [comp["vars"][0]], [comp["gate"][0]], [comp["init"][0]])
        ids_list[node_type].append(i)

    # [Pass 2] ゲート変数の収集
    current_offset = N
    for i, node_type in enumerate(nodes):
        comp = compartments[node_type]
        gate_vars = comp["vars"][1:]
        if len(gate_vars) > 0:
            gate_offsets[i] = current_offset
            acc.add(i, comp["vars"][1:], comp["gate"][1:], comp["init"][1:])
            current_offset += len(gate_vars)

    ids = defaultdict(lambda: np.array([], dtype=np.int32))
    for k, v in ids_list.items():
        ids[k] = np.array(v, dtype=np.int32)

    return {
        "gate_offsets": gate_offsets,
        "ids": ids,
        "init": acc.to_init(),
        "coords": acc.to_coords(),
    }


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


def get_gate_numpy(train_xr, target_comp_id):
    return train_xr["vars"].sel(gate=True, comp_id=target_comp_id).to_numpy()


def transform_gate(preprocessor, xr_data, target_comp_id):
    xr_gate = get_gate_numpy(xr_data, target_comp_id)
    transformed_gate = preprocessor.transform(xr_gate)
    v_soma_da = xr_data["vars"].sel(gate=False, comp_id=target_comp_id)
    new_vars = np.concatenate(
        (v_soma_da.to_numpy().reshape(-1, 1), transformed_gate), axis=1
    )

    n_latent = transformed_gate.shape[1]

    coords = StateAccumulator(
        comp_id=[target_comp_id] * (n_latent + 1),
        variable=["V"] + [f"latent{i + 1}" for i in range(n_latent)],
        gate=[False] + [True] * n_latent,
    ).to_coords()

    return set_coords(
        raw=new_vars,
        u=xr_data["I_internal"].sel(node_id=target_comp_id).to_numpy(),
        coords=coords,
        dt=float(xr_data.time[1] - xr_data.time[0]),
    )
