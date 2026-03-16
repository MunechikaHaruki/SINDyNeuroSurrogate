import numpy as np
import pysindy as ps

from neurosurrogate.modeling import neuron_core
from neurosurrogate.modeling.neuron_core import (
    HH_COST,
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
    hh_base_cost_map,
)


def make_gate_lib(funcs, names, is_product=False):
    """Gate単体、または Gate * y のペアを生成するファクトリ"""
    if not is_product:
        # 単体: lambda x: alpha_m(x)
        f_list = [f for f in funcs]
        n_list = [(lambda n: lambda x: f"{n}({x})")(n) for n in names]
    else:
        # 積: lambda x, y: alpha_m(x) * y
        f_list = [(lambda f: lambda x, y: f(x) * y)(f) for f in funcs]
        n_list = [(lambda n: lambda x, y: f"{n}({x})*{y}")(n) for n in names]
    return ps.CustomLibrary(library_functions=f_list, function_names=n_list)


gate = make_gate_lib(
    [alpha_m, alpha_h, alpha_n], ["alpha_m", "alpha_h", "alpha_n"], is_product=False
)

gate_product = make_gate_lib(
    funcs=[alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n],
    names=["alpha_m", "beta_m", "alpha_h", "beta_h", "alpha_n", "beta_n"],
    is_product=True,
)

volt_base = ps.CustomLibrary(
    library_functions=[
        lambda u, v, w: np.power(u, 3) * v * w,
        lambda u, v: np.power(u, 3) * v,
        lambda u, v: np.power(u, 4) * v,
        lambda u: np.power(u, 4),
    ],
    function_names=[
        lambda u, v, w: f"np.power({u}, 3) * {v} * {w}",
        lambda u, v: f"np.power({u}, 3) * {v}",
        lambda u, v: f"np.power({u}, 4) * {v}",
        lambda u: f"np.power({u}, 4)",
    ],
)


base = ps.CustomLibrary(
    library_functions=[lambda x: x, lambda: 1],
    function_names=[lambda x: f"{x}", lambda: "1"],
)


hh_sindy = ps.SINDy(
    feature_library=ps.GeneralizedLibrary(
        [gate, gate_product, volt_base, base],
        inputs_per_library=[  # [0,1,2]はV,g',u
            [0],
            [0, 1],
            [0, 1, 2],  # gate_product に V, m, h を渡す
            [0, 1, 2],  # base に V, m, h を渡す
        ],
    ),
    optimizer=ps.optimizers.STLSQ(threshold=0.01, normalize_columns=False, alpha=2.0),
)

MC_MODELS = {
    "hh": {
        "nodes": ["hh"],
        "edges": [],
        "stim_node": 0,
    },
    "hh3": {
        "nodes": ["passive", "hh", "passive"],
        "edges": [(0, 1, 1.0), (1, 2, 0.7)],
        "stim_node": 0,
    },
    "hh3(hhp)": {
        "nodes": ["hh", "hh", "passive"],
        "edges": [(0, 1, 1.0), (1, 2, 0.7)],
        "stim_node": 0,
    },
    "hh3(phh)": {
        "nodes": ["passive", "hh", "hh"],
        "edges": [(0, 1, 1.0), (1, 2, 0.7)],
        "stim_node": 0,
    },
    "hh5(a)": {
        "nodes": ["passive", "hh", "hh", "passive", "passive"],
        "edges": [(0, 1, 1.0), (1, 2, 0.7), (2, 3, 0.7), (3, 4, 0.5)],
        "stim_node": 0,
    },
    "hh5(b)": {
        "nodes": ["passive", "passive", "hh", "hh", "passive"],
        "edges": [(0, 1, 1.0), (1, 2, 0.7), (2, 3, 0.7), (3, 4, 0.5)],
        "stim_node": 0,
    },
    "hh5(c)": {
        "nodes": ["passive", "hh", "hh", "hh", "passive"],
        "edges": [(0, 1, 1.0), (1, 2, 0.7), (2, 3, 0.7), (3, 4, 0.5)],
        "stim_node": 0,
    },
    "hh7": {
        "nodes": ["passive", "hh", "hh", "hh", "hh", "passive", "passive"],
        "edges": [
            (0, 1, 1.0),
            (1, 2, 0.7),
            (2, 3, 0.7),
            (2, 4, 0.5),
            (3, 5, 0.5),
            (4, 6, 0.6),
        ],
        "stim_node": 0,
    },
}


COST_MAP = {
    "base": hh_base_cost_map,
    "orig": HH_COST,
}

SINDY_MODEl = {
    "sindy": hh_sindy,
    "env": neuron_core,
    "target": {
        "hh": 0,
        "hh3": 1,
        "hh3(hhp)": 1,
        "hh3(phh)": 1,
        "hh5(a)": 2,
        "hh5(b)": 2,
        "hh5(c)": 2,
        "hh7": 2,
    },
}
