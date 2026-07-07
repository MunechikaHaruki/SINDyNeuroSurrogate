from .hh import (
    HH_TEMPLATE,
    PASSIVE_TEMPLATE,
    HHParams,
    calc_hh_channel,
    calc_passive_channel,
)
from .traub import (
    TRAUB_TEMPLATE,
    TraubParams,
    calc_traub_channel,
)

COMPARTMENT_TEMPLATES = {
    "hh": HH_TEMPLATE,
    "passive": PASSIVE_TEMPLATE,
    "traub": TRAUB_TEMPLATE,
}

PARAM_CLS_BY_TYPE = {
    "hh": HHParams,
    "passive": HHParams,
    "traub": TraubParams,
}

DEFAULT_PARAMS_BY_TYPE = {
    "hh": HHParams(),
    "passive": HHParams(),
    "traub": TraubParams(),
}

__all__ = [
    "COMPARTMENT_TEMPLATES",
    "DEFAULT_PARAMS_BY_TYPE",
    "PARAM_CLS_BY_TYPE",
    "HHParams",
    "TraubParams",
    "calc_hh_channel",
    "calc_passive_channel",
    "calc_traub_channel",
]
