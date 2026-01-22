import numpy as np
import pysindy as ps
from numba import njit

from neurosurrogate.modeling.simulater import (
    alpha_h,
    alpha_m,
    beta_h,
    beta_m,
    beta_n,
)


@njit
def a_n(v):
    return 0.01 * (10 - v) / (np.exp((10 - v) / 10.0) - 1 + 0.001)  # ゼロ除算回避


@njit
def compute_theta(x0, x1, u0):
    return np.array(
        [
            alpha_m(x0),
            alpha_h(x0),
            a_n(x0),
            alpha_m(x0) * x1,
            beta_m(x0) * x1,
            alpha_h(x0) * x1,
            beta_h(x0) * x1,
            a_n(x0) * x1,
            beta_n(x0) * x1,
            (x0 * x0 * x0 * x1 * u0),
            (x0 * x0 * x0 * x1),
            (x0 * x0 * x0 * u0),
            (x1 * x1 * x1 * u0),
            (x0 * x0 * x0 * x0 * x1),
            (x0 * x0 * x0 * x0 * u0),
            (x1 * x1 * x1 * x1 * u0),
            (x0 * x0 * x0 * x0),
            (x1 * x1 * x1 * x1),
            (u0 * u0 * u0 * u0),
            x0,
            x1,
            u0,
            1,
        ]
    )


gate = ps.CustomLibrary(
    library_functions=[
        lambda x: alpha_m(x),
        lambda x: alpha_h(x),
        lambda x: a_n(x),  # ゼロ除算回避
    ],
    function_names=[
        lambda x: f"*alpha_m({x})",
        lambda x: f"*alpha_h({x})",
        lambda x: f"*a_n({x})",
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
        lambda x, y: f"*alpha_m({x})*{y}",
        lambda x, y: f"*beta_m({x})*{y}",
        lambda x, y: f"*alpha_h({x})*{y}",
        lambda x, y: f"*beta_h({x})*{y}",
        lambda x, y: f"*a_n({x})*{y}",
        lambda x, y: f"*beta_n({x})*{y}",
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
        lambda u, v, w: f"*({u}*{u}*{u}*{v}*{w})",
        lambda u, v: f"*({u}*{u}*{u}*{v})",
        lambda u, v: f"*({u}*{u}*{u}*{u}*{v})",
        lambda u: f"*({u}*{u}*{u}*{u})",
    ],
)

base = ps.CustomLibrary(
    library_functions=[lambda x: x, lambda: 1],
    function_names=[lambda x: f"*{x}", lambda: "*1"],
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
input_features = ["v", "g", "u"]