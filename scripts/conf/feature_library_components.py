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


def make_gate_lib(funcs, is_product=False):
    """Gate単体、または Gate * y のペアを生成するファクトリ"""
    f_names = [f.__name__ for f in funcs]
    if not is_product:
        # 単体: lambda x: alpha_m(x)
        f_list = [f for f in funcs]
        n_list = [(lambda n: lambda x: f"{n}({x})")(n) for n in f_names]
    else:
        # 積: lambda x, y: alpha_m(x) * y
        f_list = [(lambda f: lambda x, y: f(x) * y)(f) for f in funcs]
        n_list = [(lambda n: lambda x, y: f"{n}({x})*{y}")(n) for n in f_names]
    return ps.CustomLibrary(library_functions=f_list, function_names=n_list)


def make_volt_lib(specs):
    """(累乗, 変数個数) のタプルリストから生成"""
    f_list, n_list = [], []

    # 1. 内部で「関数を作るための関数」を定義（pを固定するため）
    def create_u_p_v_w(p_val):
        return (
            lambda u, v, w: np.power(u, p_val) * v * w,
            lambda u, v, w: f"np.power({u}, {p_val}) * {v} * {w}",
        )

    def create_u_p_v(p_val):
        return (
            lambda u, v: np.power(u, p_val) * v,
            lambda u, v: f"np.power({u}, {p_val}) * {v}",
        )

    def create_u_p(p_val):
        return lambda u: np.power(u, p_val), lambda u: f"np.power({u}, {p_val})"

    # 2. ループで適切な関数を生成して追加
    for p, vars_count in specs:
        if vars_count == 2:
            f, n = create_u_p_v_w(p)
        elif vars_count == 1:
            f, n = create_u_p_v(p)
        else:
            f, n = create_u_p(p)

        f_list.append(f)
        n_list.append(n)

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


base_lib = ps.CustomLibrary(
    library_functions=[lambda x: x, lambda: 1],
    function_names=[lambda x: f"{x}", lambda: "1"],
)


FUNC_REGISTRY = {
    "alpha_m": alpha_m,
    "alpha_h": alpha_h,
    "alpha_n": alpha_n,
    "beta_m": beta_m,
    "beta_h": beta_h,
    "beta_n": beta_n,
}


LIB_BUILDER_REGISTRY = {
    "gate": lambda spec: make_gate_lib(
        [FUNC_REGISTRY[f] for f in spec["funcs"]],
        is_product=spec.get("is_product", False),
    ),
    "volt": lambda spec: make_volt_lib([tuple(s) for s in spec["specs"]]),
    "base": lambda spec: base_lib,
    "gate_poly_volt": lambda spec: make_gate_poly_volt_lib(spec["max_power"]),
}
