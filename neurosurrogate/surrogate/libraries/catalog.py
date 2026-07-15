from collections.abc import Callable

import sympy as sp

from ...compartments.hh import (
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
)
from .entry import LibraryEntry

# ---------------------------------------------------------------------------
# カタログ層。項カタログ (式リスト) が唯一の真実源。HH レート関数は数値安定な
# exp 実装を壊さないため未定義関数シンボルで構造だけ持ち、lambdify 時に注入する。
# ---------------------------------------------------------------------------

_RATE_IMPL: dict[str, Callable] = {
    "alpha_m": alpha_m,
    "beta_m": beta_m,
    "alpha_h": alpha_h,
    "beta_h": beta_h,
    "alpha_n": alpha_n,
    "beta_n": beta_n,
}
_RATE_S: dict[str, sp.Function] = {nm: sp.Function(nm) for nm in _RATE_IMPL}

V, G = sp.symbols("V g")  # 電位 / ゲート (product・relaxation・gate_poly 用)
M, H, N = sp.symbols("m h n")  # volt 項の 3 ゲート


def _entry(expr: sp.Expr, *args: sp.Symbol) -> LibraryEntry:
    return LibraryEntry(
        expr=expr,
        args=args,
        func=sp.lambdify(args, expr, modules=[_RATE_IMPL, "jax"]),
    )


_GATE_RATES = ["alpha_m", "beta_m", "alpha_h", "beta_h", "alpha_n", "beta_n"]
_FORWARD_RATES = ["alpha_m", "alpha_h", "alpha_n"]
_RELAX_PAIRS = [("alpha_m", "beta_m"), ("alpha_h", "beta_h"), ("alpha_n", "beta_n")]


def _basis(n: int) -> list[LibraryEntry]:
    """n 入力の射影 (各引数そのまま) + 定数 1。"""
    xs = tuple(sp.Symbol(f"x{i}") for i in range(n))
    return [_entry(x, *xs) for x in xs] + [_entry(sp.Integer(1), *xs)]


FIXED_LIB_ENTRIES: dict[str, list[LibraryEntry]] = {
    "hh_gate": [_entry(_RATE_S[nm](V), V) for nm in _GATE_RATES],
    "hh_gate_product": [_entry(_RATE_S[nm](V) * G, V, G) for nm in _GATE_RATES],
    "hh_gate_forward": [_entry(_RATE_S[nm](V), V) for nm in _FORWARD_RATES],
    "hh_gate_forward_product": [
        _entry(_RATE_S[nm](V) * G, V, G) for nm in _FORWARD_RATES
    ],
    "hh_relaxation_driver": [_entry(_RATE_S[a](V), V) for a, _ in _RELAX_PAIRS],
    "hh_relaxation_decay": [
        _entry((_RATE_S[a](V) + _RATE_S[b](V)) * G, V, G) for a, b in _RELAX_PAIRS
    ],
    "volt": [_entry(e, M, H, N) for e in (M**3 * H * N, M**3 * H, M**4 * H, M**4)],
    "gate_poly_volt": [
        _entry(expr, V, G) for p in range(1, 5) for expr in (G**p, G**p * V)
    ],
}

VARIADIC_LIB_ENTRIES: dict[str, Callable[[int], list[LibraryEntry]]] = {
    "basis": _basis,
}
