"""サロゲート ansatz 共通ヘルパ: ゲート抽出と preprocessor による latent 変換。

ansatz (sindy/hybrid/...) が学習・診断で共有する。特定 ansatz に依存しない。
"""

from typing import Any

import numpy as np
import xarray as xr

from ...core import access
from ...core.coords import StateAccumulator, set_coords


def get_gate_numpy(train_xr: xr.Dataset, target_comp_id: int) -> np.ndarray:
    return access.gate_matrix(train_xr, target_comp_id)


def transform_gate(
    preprocessor: Any, xr_data: xr.Dataset, target_comp_id: int
) -> xr.Dataset:
    transformed_gate = preprocessor.transform(get_gate_numpy(xr_data, target_comp_id))
    n_latent = transformed_gate.shape[1]

    return set_coords(
        raw=np.concatenate(
            (
                access.potential(xr_data, target_comp_id).reshape(-1, 1),
                transformed_gate,
            ),
            axis=1,
        ),
        u=access.i_internal_values(xr_data, target_comp_id),
        coords=StateAccumulator(
            comp_id=[target_comp_id] * (n_latent + 1),
            variable=["V"] + [f"latent{i + 1}" for i in range(n_latent)],
            gate=[False] + [True] * n_latent,
        ).to_coords(),
        dt=float(xr_data.time[1] - xr_data.time[0]),
    )
