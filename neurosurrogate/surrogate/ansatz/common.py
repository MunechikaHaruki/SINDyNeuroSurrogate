"""サロゲート ansatz 共通ヘルパ: ゲート抽出と preprocessor による latent 変換。

ansatz (sindy/hybrid/...) が学習・診断で共有する。特定 ansatz に依存しない。
"""

from typing import Any

import xarray as xr

from ...core import access
from ...core.coords import set_latent_coords


def transform_gate(
    preprocessor: Any, xr_data: xr.Dataset, target_comp_id: int
) -> xr.Dataset:
    return set_latent_coords(
        v=access.potential(xr_data, target_comp_id),
        latent=preprocessor.transform(access.gate_matrix(xr_data, target_comp_id)),
        u=access.i_internal_values(xr_data, target_comp_id),
        comp_id=target_comp_id,
        dt=access.dt(xr_data),
    )
