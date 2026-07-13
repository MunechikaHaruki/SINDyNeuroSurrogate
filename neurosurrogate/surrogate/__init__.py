from .base import NeuroSurrogateBase, SurrogateMeta, get_gate_numpy, transform_gate
from .hybrid import HybridSINDyNeuroSurrogate
from .sindy import SINDyNeuroSurrogate

SURR_CLS: dict[str, type[NeuroSurrogateBase]] = {
    cls.SURROGATE_TYPE: cls for cls in (SINDyNeuroSurrogate, HybridSINDyNeuroSurrogate)
}

__all__ = [
    "SURR_CLS",
    "HybridSINDyNeuroSurrogate",
    "NeuroSurrogateBase",
    "SINDyNeuroSurrogate",
    "SurrogateMeta",
    "get_gate_numpy",
    "transform_gate",
]
