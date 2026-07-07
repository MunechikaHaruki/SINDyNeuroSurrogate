from .hh import (
    HH_TEMPLATE,
    HH_TYPE,
    PASSIVE_TEMPLATE,
    PASSIVE_TYPE,
    HHParams,
    PassiveParams,
)
from .traub import (
    TRAUB_TEMPLATE,
    TRAUB_TYPE,
    TraubParams,
)

# CompartmentType (物理の型) を name で引く
COMPARTMENT_TYPES = {
    "hh": HH_TYPE,
    "passive": PASSIVE_TYPE,
    "traub": TRAUB_TYPE,
}

# 後方互換: 空 name の Compartment テンプレ
COMPARTMENT_TEMPLATES = {
    "hh": HH_TEMPLATE,
    "passive": PASSIVE_TEMPLATE,
    "traub": TRAUB_TEMPLATE,
}

__all__ = [
    "COMPARTMENT_TEMPLATES",
    "COMPARTMENT_TYPES",
    "HHParams",
    "PassiveParams",
    "TraubParams",
]
