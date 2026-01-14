import numpy as np
import pysindy as ps
from numba import njit


@njit
def a_m(v):
    return 0.1 * (25 - v) / (np.exp((25 - v) / 10.0) - 1)


@njit
def b_m(v):
    return 4.0 * np.exp(-v / 18.0)


@njit
def a_h(v):
    return 0.07 * np.exp(-v / 20.0)


@njit
def b_h(v):
    return 1.0 / (np.exp((30 - v) / 10.0) + 1)


@njit
def a_n(v):
    return 0.01 * (10 - v) / (np.exp((10 - v) / 10.0) - 1 + 0.001)  # ゼロ除算回避


@njit
def b_n(v):
    return 0.125 * np.exp(-v / 80.0)


@njit
def compute_theta(x0, x1, u0):
    return np.array(
        [
            a_m(x0),
            a_h(x0),
            a_n(x0),
            a_m(x0) * x1,
            b_m(x0) * x1,
            a_h(x0) * x1,
            b_h(x0) * x1,
            a_n(x0) * x1,
            b_n(x0) * x1,
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
        lambda x: a_m(x),
        lambda x: a_h(x),
        lambda x: a_n(x),  # ゼロ除算回避
    ],
    function_names=[
        lambda x: f"*a_m({x})",
        lambda x: f"*a_h({x})",
        lambda x: f"*a_n({x})",
    ],
)

gate_product = ps.CustomLibrary(
    library_functions=[
        lambda x, y: a_m(x) * y,
        lambda x, y: b_m(x) * y,
        lambda x, y: a_h(x) * y,
        lambda x, y: b_h(x) * y,
        lambda x, y: a_n(x) * y,
        lambda x, y: b_n(x) * y,
    ],
    function_names=[
        lambda x, y: f"*a_m({x})*{y}",
        lambda x, y: f"*b_m({x})*{y}",
        lambda x, y: f"*a_h({x})*{y}",
        lambda x, y: f"*b_h({x})*{y}",
        lambda x, y: f"*a_n({x})*{y}",
        lambda x, y: f"*b_n({x})*{y}",
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


def base_hh():
    """
    GeneralizedLibrary を使用して、各ライブラリが参照する変数の列（インデックス）を指定します。
    仮定: 列0=V, 列1=m, 列2=h, 列3=n
    """
    return ps.GeneralizedLibrary(
        [gate, gate_product, volt_base, base],
        inputs_per_library=[
            [0],
            [0, 1],
            [0, 1, 2],  # gate_product に V, m, h を渡す
            [0, 1, 2],  # base に V, m, h を渡す
        ],
    )
