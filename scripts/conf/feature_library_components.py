from dataclasses import dataclass
from typing import Callable

import numpy as np

from neurosurrogate.calc_utils import OpCost
from neurosurrogate.neuron_core import (
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
    njit,  # noqa: F401
)
from neurosurrogate.profiler_model import HH_RATE_COST_MAP

FUNC_REGISTRY = {
    "alpha_m": alpha_m,
    "alpha_h": alpha_h,
    "alpha_n": alpha_n,
    "beta_m": beta_m,
    "beta_h": beta_h,
    "beta_n": beta_n,
}

GATE_PAIR_REGISTRY = {
    "m": (alpha_m, beta_m),
    "h": (alpha_h, beta_h),
    "n": (alpha_n, beta_n),
}


@dataclass(frozen=True)
class LibraryEntry:
    func: Callable
    name_func: Callable
    cost: OpCost


def make_gate_lib(func_names, is_product=False):
    """Gate 単体、または Gate * y のペア。"""
    entries: list[LibraryEntry] = []
    for name in func_names:
        rate_cost = HH_RATE_COST_MAP[name]
        if not is_product:
            f = FUNC_REGISTRY[name]
            n = (lambda nm: lambda x: f"{nm}({x})")(name)
            entries.append(LibraryEntry(func=f, name_func=n, cost=rate_cost))
        else:
            f = (lambda fn: lambda x, y: fn(x) * y)(FUNC_REGISTRY[name])
            n = (lambda nm: lambda x, y: f"{nm}({x})*{y}")(name)
            # alpha_k(V) を計算して g0 と乗算 → 内側関数 + mul 1 回
            entries.append(
                LibraryEntry(func=f, name_func=n, cost=rate_cost + OpCost(mul=1))
            )
    return entries


def make_relaxation_1var_lib(gates_list):
    """緩和ダイナミクスの駆動項: alpha_k(V)。inputs=[0] (V のみ)。"""
    entries: list[LibraryEntry] = []
    for g in gates_list:
        alpha_f, _ = GATE_PAIR_REGISTRY[g]
        a_name = alpha_f.__name__
        entries.append(
            LibraryEntry(
                func=(lambda af: lambda x: af(x))(alpha_f),
                name_func=(lambda an: lambda x: f"{an}({x})")(a_name),
                cost=HH_RATE_COST_MAP[a_name],
            )
        )
    return entries


def make_relaxation_2var_lib(gates_list):
    """緩和ダイナミクスの減衰項: (alpha_k + beta_k)(V) * g0。inputs=[0, 1] (V, g0)。"""
    entries: list[LibraryEntry] = []
    for g in gates_list:
        alpha_f, beta_f = GATE_PAIR_REGISTRY[g]
        a_name, b_name = alpha_f.__name__, beta_f.__name__
        # alpha_k(V) + beta_k(V) → pm 1, それを g0 と乗算 → mul 1
        compose_extra = OpCost(pm=1, mul=1)
        entries.append(
            LibraryEntry(
                func=(lambda af, bf: lambda x, y: (af(x) + bf(x)) * y)(alpha_f, beta_f),
                name_func=(lambda an, bn: lambda x, y: f"({an}({x})+{bn}({x}))*{y}")(
                    a_name, b_name
                ),
                cost=HH_RATE_COST_MAP[a_name]
                + HH_RATE_COST_MAP[b_name]
                + compose_extra,
            )
        )
    return entries


def make_volt_lib():
    """V^a * g0^b * (...) 系の多項式項 (legacy)。"""
    entries = [
        LibraryEntry(
            func=lambda u, v, w: np.power(u, 3) * v * w,
            name_func=lambda u, v, w: f"np.power({u}, 3) * {v} * {w}",
            # u^3 = mul 2, *v = mul 1, *w = mul 1
            cost=OpCost(mul=4),
        ),
        LibraryEntry(
            func=lambda u, v, w: np.power(u, 3) * v,
            name_func=lambda u, v, w: f"np.power({u}, 3) * {v}",
            cost=OpCost(mul=3),
        ),
        LibraryEntry(
            func=lambda u, v, w: np.power(u, 4) * v,
            name_func=lambda u, v, w: f"np.power({u}, 4) * {v}",
            cost=OpCost(mul=4),
        ),
        LibraryEntry(
            func=lambda u, v, w: np.power(u, 4),
            name_func=lambda u, v, w: f"np.power({u}, 4)",
            cost=OpCost(mul=3),
        ),
    ]
    return entries


def make_gate_poly_volt_lib(max_power: int):
    """g^k と g^k * V (k=1..max_power) のペア。inputs 順は (V, g) で固定。"""
    entries: list[LibraryEntry] = []
    for p in range(1, max_power + 1):
        # g^p
        entries.append(
            LibraryEntry(
                func=(lambda pp: lambda V, g: np.power(g, pp))(p),
                name_func=(lambda pp: lambda V, g: f"np.power({g}, {pp})")(p),
                cost=OpCost(mul=max(0, p - 1)),
            )
        )
        # g^p * V
        entries.append(
            LibraryEntry(
                func=(lambda pp: lambda V, g: np.power(g, pp) * V)(p),
                name_func=(lambda pp: lambda V, g: f"np.power({g}, {pp}) * {V}")(p),
                cost=OpCost(mul=max(0, p - 1) + 1),
            )
        )
    return entries


def make_identity_lib():
    """variable をそのまま渡す。zero cost。"""
    entries = [
        LibraryEntry(
            func=lambda x: x,
            name_func=lambda x: f"{x}",
            cost=OpCost(),
        )
    ]
    return entries


def make_const_lib():
    """定数項 1。zero cost。"""
    entries = [
        LibraryEntry(
            func=lambda: 1,
            name_func=lambda: "1",
            cost=OpCost(),
        )
    ]
    return entries


LIB_BUILDER_REGISTRY = {
    "gate": lambda spec: make_gate_lib(
        func_names=spec["funcs"],
        is_product=spec.get("is_product", False),
    ),
    "volt": lambda spec: make_volt_lib(),
    "gate_poly_volt": lambda spec: make_gate_poly_volt_lib(spec["max_power"]),
    "identity": lambda spec: make_identity_lib(),
    "const": lambda spec: make_const_lib(),
    "relaxation1": lambda spec: make_relaxation_1var_lib(spec["gates"]),
    "relaxation2": lambda spec: make_relaxation_2var_lib(spec["gates"]),
}
