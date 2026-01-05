import numpy as np
import pysindy as ps


def a_m(v):
    return 0.1 * (25 - v) / (np.exp((25 - v) / 10.0) - 1)


def b_m(v):
    return 4.0 * np.exp(-v / 18.0)


def a_h(v):
    return 0.07 * np.exp(-v / 20.0)


def b_h(v):
    return 1.0 / (np.exp((30 - v) / 10.0) + 1)


def a_n(v):
    return 0.01 * (10 - v) / (np.exp((10 - v) / 10.0) - 1 + 0.001)  # ゼロ除算回避


def b_n(v):
    return 0.125 * np.exp(-v / 80.0)


def derivative_gate(alpha, beta, gate):
    return alpha * (1 - gate) - (beta * gate)


def derivative_m(cls, v, m):
    return cls.derivative_gate(cls.a_m(v), cls.b_m(v), m)


def derivative_h(cls, v, h):
    return cls.derivative_gate(cls.a_h(v), cls.b_h(v), h)


def derivative_n(cls, v, n):
    return cls.derivative_gate(cls.a_n(v), cls.b_n(v), n)


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
