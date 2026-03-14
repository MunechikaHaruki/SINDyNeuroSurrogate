import sys

import numpy as np
import pysindy as ps
from numba import njit

from neurosurrogate.modeling.neuron_core import (
    alpha_h,
    alpha_m,
    beta_h,
    beta_m,
    beta_n,
)

cost_map = {
    "alpha_m": {
        "exp": 1,
        "div": 1,
        "pm": 2,  # 2.5 - 0.1v (1回分) と 分母の - 1.0
        "mul": 2,  # 0.1 * v (1回分)
    },
    "beta_m": {
        "exp": 1,
        "div": 1,
        "pm": 1,  # -v
        "mul": 1,  # 4.0 * exp
    },
    "alpha_h": {
        "exp": 1,
        "div": 1,
        "pm": 1,
        "mul": 1,
    },
    "beta_h": {
        "exp": 1,
        "div": 1,
        "pm": 2,  # 3.0 - 0.1v と + 1.0
        "mul": 1,  # 0.1 * v
    },
    "alpha_n": {
        "exp": 1,
        "div": 1,
        "pm": 2,
        "mul": 2,
    },
    "beta_n": {
        "exp": 1,
        "div": 1,
        "pm": 1,
        "mul": 1,
    },
    "a_n": {"exp": 1, "div": 1, "pm": 2, "mul": 2},
}


@njit
def a_n(v):
    return 0.01 * (10 - v) / (np.exp((10 - v) / 10.0) - 1 + 0.001)  # ゼロ除算回避


gate = ps.CustomLibrary(
    library_functions=[
        lambda x: alpha_m(x),
        lambda x: alpha_h(x),
        lambda x: a_n(x),  # ゼロ除算回避
    ],
    function_names=[
        lambda x: f"alpha_m({x})",
        lambda x: f"alpha_h({x})",
        lambda x: f"a_n({x})",
    ],
)

gate_product = ps.CustomLibrary(
    library_functions=[
        lambda x, y: alpha_m(x) * y,
        lambda x, y: beta_m(x) * y,
        lambda x, y: alpha_h(x) * y,
        lambda x, y: beta_h(x) * y,
        lambda x, y: a_n(x) * y,
        lambda x, y: beta_n(x) * y,
    ],
    function_names=[
        lambda x, y: f"alpha_m({x})*{y}",
        lambda x, y: f"beta_m({x})*{y}",
        lambda x, y: f"alpha_h({x})*{y}",
        lambda x, y: f"beta_h({x})*{y}",
        lambda x, y: f"a_n({x})*{y}",
        lambda x, y: f"beta_n({x})*{y}",
    ],
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
        inputs_per_library=[
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

SINDY_MODEl = {
    "sindy": hh_sindy,
    "env": sys.modules[__name__],
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
