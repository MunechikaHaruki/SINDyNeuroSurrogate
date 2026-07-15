from collections.abc import Callable
from dataclasses import dataclass, field

import pysindy as ps
import sympy as sp
from sympy.core.function import AppliedUndef

from ..compartments.hh import (
    HH_RATE_COST_MAP,
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
)
from ..core.opcost import OpCost

# ---------------------------------------------------------------------------
# 項 = 1つの sympy 式。func(lambdify) / name(render) / cost(op_cost) / arity は
# すべてここから派生する (手書き3重同期を廃止)。HH レート関数は数値安定な exp
# 実装を壊さないため未定義関数シンボルで構造だけ保持し、lambdify 時に注入する。
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


def op_cost(e: sp.Expr) -> OpCost:
    """sympy 式木を辿り OpCost へ集計。冪→mul(p-1)、積→mul(項数-1)、和→pm(項数-1)、
    未定義レート関数→HH_RATE_COST_MAP。"""
    if e.is_Symbol or e.is_Number:
        return OpCost()
    if isinstance(e, AppliedUndef):
        return sum((op_cost(a) for a in e.args), HH_RATE_COST_MAP[e.func.__name__])
    if isinstance(e, sp.Pow):
        base, exp = e.args
        if not (exp.is_Integer and int(exp) >= 1):
            raise ValueError(f"整数冪 (>=1) のみ対応: {e}")
        return op_cost(base) + OpCost(mul=int(exp) - 1)
    if isinstance(e, sp.Mul):
        return sum((op_cost(a) for a in e.args), OpCost(mul=len(e.args) - 1))
    if isinstance(e, sp.Add):
        return sum((op_cost(a) for a in e.args), OpCost(pm=len(e.args) - 1))
    raise ValueError(f"op_cost 未対応ノード: {e!r}")


@dataclass(frozen=True)
class LibraryEntry:
    """候補項1つ。expr が真実源、args は位置引数シンボル (arity 兼束縛順)。"""

    expr: sp.Expr
    args: tuple[sp.Symbol, ...]
    func: Callable = field(compare=False)  # lambdify 結果 (jax-ready、束縛時1回)

    @property
    def cost(self) -> OpCost:
        return op_cost(self.expr)

    @property
    def arity(self) -> int:
        return len(self.args)

    def name_func(self, *names: str) -> str:
        """位置引数名を代入した文字列 (pysindy の function_names / コスト表 共通源)。"""
        return str(
            self.expr.subs(dict(zip(self.args, map(sp.Symbol, names), strict=True)))
        )

    def to_cost_entry(self, input_names: list[str]) -> tuple[str, OpCost]:
        return self.name_func(*input_names), self.cost


def _entry(expr: sp.Expr, *args: sp.Symbol) -> LibraryEntry:
    return LibraryEntry(
        expr=expr,
        args=args,
        func=sp.lambdify(args, expr, modules=[_RATE_IMPL, "jax"]),
    )


@dataclass(frozen=True)
class SubLibrary:
    """1 library_spec 単位。entries + 入力インデックス binding。"""

    entries: list[LibraryEntry]
    inputs: list[int]

    def to_ps_library(self) -> ps.CustomLibrary:
        return ps.CustomLibrary(
            library_functions=[e.func for e in self.entries],
            function_names=[e.name_func for e in self.entries],
        )

    def to_cost_dict(self, input_names: list[str]) -> dict[str, OpCost]:
        bound = [input_names[i] for i in self.inputs]
        return dict(e.to_cost_entry(bound) for e in self.entries)


@dataclass(frozen=True)
class FeatureLibrary:
    sub_libraries: list[SubLibrary]
    library: ps.GeneralizedLibrary

    def to_base_cost(self, input_names: list[str]) -> dict[str, OpCost]:
        base_cost: dict[str, OpCost] = {}
        for sl in self.sub_libraries:
            new_data = sl.to_cost_dict(input_names)
            if dup := base_cost.keys() & new_data.keys():
                raise KeyError(f"library間で feature名 重複: {sorted(dup)}")
            base_cost |= new_data
        return base_cost

    @staticmethod
    def build(library_specs: list[dict]) -> "FeatureLibrary":
        def _resolve(spec: dict) -> SubLibrary:
            t = spec["type"]
            inputs = spec["inputs"]
            if t in FIXED_LIB_ENTRIES:
                entries = FIXED_LIB_ENTRIES[t]
                if len(inputs) != entries[0].arity:
                    raise ValueError(
                        f"type={t!r} は arity={entries[0].arity} 要求、"
                        f"inputs={inputs} (arity={len(inputs)})"
                    )
                return SubLibrary(entries=entries, inputs=inputs)
            if t in VARIADIC_LIB_ENTRIES:
                return SubLibrary(
                    entries=VARIADIC_LIB_ENTRIES[t](len(inputs)), inputs=inputs
                )
            known = sorted(
                list(FIXED_LIB_ENTRIES.keys()) + list(VARIADIC_LIB_ENTRIES.keys())
            )
            raise ValueError(f"未知 library type: {t!r}。対応 type: {known}")

        subs = [_resolve(s) for s in library_specs]
        return FeatureLibrary(
            sub_libraries=subs,
            library=ps.GeneralizedLibrary(
                [sl.to_ps_library() for sl in subs],
                inputs_per_library=[sl.inputs for sl in subs],
            ),
        )


# ---------------------------------------------------------------------------
# 項カタログ。式リストが唯一の真実源 → func/name/cost/arity は自動派生。
# ---------------------------------------------------------------------------

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
