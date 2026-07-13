from pathlib import Path

import joblib

from .base import (
    BUNDLE_FILE,
    NeuroSurrogateBase,
    SurrogateMeta,
    get_gate_numpy,
    transform_gate,
)
from .hybrid import HybridSINDyNeuroSurrogate
from .sindy import SINDyNeuroSurrogate

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
    "transform_gate",
]
