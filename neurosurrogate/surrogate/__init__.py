from .base import NeuroSurrogateBase, get_gate_numpy, transform_gate
from .hybrid import HybridSINDyNeuroSurrogate
from .sindy import SINDyNeuroSurrogate

SURR_CLS: dict[str, type[NeuroSurrogateBase]] = {
    "sindy": SINDyNeuroSurrogate,
    "hybrid": HybridSINDyNeuroSurrogate,
}

__all__ = [
    "SURR_CLS",
    "HybridSINDyNeuroSurrogate",
    "NeuroSurrogateBase",
    "SINDyNeuroSurrogate",
    "get_gate_numpy",
    "transform_gate",
]
