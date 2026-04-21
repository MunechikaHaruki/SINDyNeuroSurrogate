import numpy as np
import pysindy as ps

from neurosurrogate.modeling.neuron_core import (
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
)

FUNC_REGISTRY = {
    "alpha_m": alpha_m,
    "alpha_h": alpha_h,
    "alpha_n": alpha_n,
    "beta_m": beta_m,
    "beta_h": beta_h,
    "beta_n": beta_n,
}


def make_gate_lib(func_names, is_product=False):
    """Gate単体、または Gate * y のペアを生成するファクトリ"""
    if not is_product:
        # 単体: lambda x: alpha_m(x)
        f_list = [FUNC_REGISTRY[name] for name in func_names]
        n_list = [(lambda n: lambda x: f"{n}({x})")(n) for n in func_names]
    else:
        # 積: lambda x, y: alpha_m(x) * y
        f_list = [
            (lambda f: lambda x, y: f(x) * y)(FUNC_REGISTRY[name])
            for name in func_names
        ]
        n_list = [(lambda n: lambda x, y: f"{n}({x})*{y}")(n) for n in func_names]
    return ps.CustomLibrary(library_functions=f_list, function_names=n_list)


def make_volt_lib():
    f_list = [
        lambda u, v, w: np.power(u, 3) * v * w,
        lambda u, v: np.power(u, 3) * v,
        lambda u, v: np.power(u, 4) * v,
        lambda u: np.power(u, 4),
    ]
    n_list = [
        lambda u, v, w: f"np.power({u}, 3) * {v} * {w}",
        lambda u, v: f"np.power({u}, 3) * {v}",
        lambda u, v: f"np.power({u}, 4) * {v}",
        lambda u: f"np.power({u}, 4)",
    ]
    return ps.CustomLibrary(library_functions=f_list, function_names=n_list)


def make_gate_poly_volt_lib(max_power: int):
    """
    V の式向け: g の k=1..max_power 次多項式と、それ × V のペアを生成。
    inputs 順は (V, g) で固定。
    """
    f_list, n_list = [], []

    def create_poly(p):
        return (
            lambda V, g: np.power(g, p),
            lambda V, g: f"np.power({g}, {p})",
        )

    def create_poly_V(p):
        return (
            lambda V, g: np.power(g, p) * V,
            lambda V, g: f"np.power({g}, {p}) * {V}",
        )

    for p in range(1, max_power + 1):
        f, n = create_poly(p)
        f_list.append(f)
        n_list.append(n)
        f, n = create_poly_V(p)
        f_list.append(f)
        n_list.append(n)

    return ps.CustomLibrary(library_functions=f_list, function_names=n_list)


LIB_BUILDER_REGISTRY = {
    "gate": lambda spec: make_gate_lib(
        func_names=spec["funcs"],
        is_product=spec.get("is_product", False),
    ),
    "volt": lambda spec: make_volt_lib(),
    "gate_poly_volt": lambda spec: make_gate_poly_volt_lib(spec["max_power"]),
    "identity": lambda spec: ps.IdentityLibrary(),
    "const": lambda spec: ps.PolynomialLibrary(degree=0, include_bias=True),
}
