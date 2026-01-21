from .simulater import (
    HH_Params_numba,
    ThreeComp_Params_numba,
    hh3_simulate_numba_wrapper,
    hh_simulate_numba_wrapper,
)

PARAMS_REGISTRY = {
    "hh": HH_Params_numba,
    "hh3": ThreeComp_Params_numba,
    "hh_numba": HH_Params_numba,
    "hh3_numba": ThreeComp_Params_numba,
}
SIMULATOR_REGISTRY = {
    "hh": hh_simulate_numba_wrapper,
    "hh3": hh3_simulate_numba_wrapper,
    "hh_numba": hh_simulate_numba_wrapper,
    "hh3_numba": hh3_simulate_numba_wrapper,
}
