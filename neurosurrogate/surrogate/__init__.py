from pathlib import Path

import joblib
import numpy as np

from ..core import access
from ..core.coords import StateAccumulator, set_coords
from .base import (
    BUNDLE_FILE,
    NeuroSurrogateBase,
    SurrogateMeta,
)
from .replace import Verdict, verdict


def get_gate_numpy(train_xr, target_comp_id):
    return access.gate_matrix(train_xr, target_comp_id)


def transform_gate(preprocessor, xr_data, target_comp_id):
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
    "Verdict",
    "get_gate_numpy",
    "load_surrogate",
    "transform_gate",
    "verdict",
]
