from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xarray as xr

from ..core import access
from ..core.coords import StateAccumulator, set_coords
from ..core.network import DatasetConfig
from .base import (
    BUNDLE_FILE,
    NeuroSurrogateBase,
    SurrogateMeta,
)
from .replace import Verdict, verdict


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


def preprocessed_latent(
    surrogate: NeuroSurrogateBase,
    dataset: DatasetConfig,
    sim_ds: xr.Dataset,
    comp_id: int,
) -> xr.Dataset:
    """comp_id ノードのゲートを preprocessor で latent 圧縮した (V, latent...) xr を返す
    (診断用)。学習ドメイン外 (verdict != REPLACE) は latent 比較不可でエラー化。"""
    comp = dataset.net.nodes[comp_id]
    if (v := verdict(surrogate.meta, comp)) is not Verdict.REPLACE:
        raise ValueError(
            f"comp {comp.name!r} は学習ドメイン外 ({v.name}) → latent 比較不可 "
            f"(学習型 {surrogate.meta.train_comp_type.name!r})"
        )
    return transform_gate(surrogate.preprocessor_bundle.preprocessor, sim_ds, comp_id)


# get_gate_numpy/transform_gate は hybrid/sindy が使うため、
# それらの import より前に定義しておく (親 __init__ 経由の循環回避)
from .hybrid import HybridSINDyNeuroSurrogate  # noqa: E402
from .sindy import SINDyNeuroSurrogate  # noqa: E402

SURR_CLS: dict[str, type[NeuroSurrogateBase]] = {
    cls.SURROGATE_TYPE: cls for cls in (SINDyNeuroSurrogate, HybridSINDyNeuroSurrogate)
}


def load_surrogate(dir: Path | str) -> NeuroSurrogateBase:
    data = joblib.load(Path(dir) / BUNDLE_FILE)
    meta: SurrogateMeta = data["meta"]
    cls = SURR_CLS[meta.surrogate_type]
    surrogate = cls.__new__(cls)
    surrogate._meta = meta
    surrogate._set_bundles(
        sindy_bundle=data["sindy_bundle"],
        preprocessor_bundle=data["preprocessor_bundle"],
    )
    return surrogate


__all__ = [
    "SURR_CLS",
    "HybridSINDyNeuroSurrogate",
    "NeuroSurrogateBase",
    "SINDyNeuroSurrogate",
    "SurrogateMeta",
    "get_gate_numpy",
    "load_surrogate",
    "preprocessed_latent",
    "transform_gate",
]
