from .hh import hh_simulator

PARAMS_REGISTRY = {"hh": hh_simulator.HH_Params, "hh3": hh_simulator.ThreeComp_Params}
SIMULATOR_REGISTRY = {
    "hh": hh_simulator.hh_simulate,
    "hh3": hh_simulator.threecomp_simulate,
}
