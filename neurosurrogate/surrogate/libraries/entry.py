from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pysindy as ps
import sympy as sp
from sympy.core.function import AppliedUndef

from ...compartments.hh import HH_RATE_COST_MAP
from ...core.opcost import OpCost

if TYPE_CHECKING:
    from ..ansatz.roles import Roles

# ---------------------------------------------------------------------------
# ロジック層。項 = 1つの sympy 式。func(lambdify) / name(subs) / cost(op_cost) /
# arity(len args) はすべてここから派生する。項カタログは catalog.py。
# ---------------------------------------------------------------------------


def op_cost(e: sp.Expr) -> OpCost:
    """式木の総 OpCost = 自ノードのコスト + 全子の総和 (再帰)。"""
    return sum((op_cost(child) for child in e.args), _node_cost(e))


def _node_cost(e: sp.Expr) -> OpCost:
    """子を除いた自ノード単独の演算コスト。冪→mul(p-1)、積→mul(項数-1)、
    和→pm(項数-1)、未定義レート関数→HH_RATE_COST_MAP、葉(記号/数)→0。"""
    match e:
        case _ if e.is_Symbol or e.is_Number:
            return OpCost()
        case AppliedUndef():
            return HH_RATE_COST_MAP[e.func.__name__]
        case sp.Pow() if e.exp.is_Integer and int(e.exp) >= 1:
            return OpCost(mul=int(e.exp) - 1)
        case sp.Pow():
            raise ValueError(f"整数冪 (>=1) のみ対応: {e}")
        case sp.Mul():
            return OpCost(mul=len(e.args) - 1)
        case sp.Add():
            return OpCost(pm=len(e.args) - 1)
        case _:
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


@dataclass(frozen=True)
class SubLibrary:
    """1 library_spec を解決した 1 束。entries + 入力列 binding。"""

    entries: list[LibraryEntry]
    inputs: list[int]

    @classmethod
    def expand(cls, spec: dict, roles: "Roles", n_inputs: int) -> list["SubLibrary"]:
        """1 library_spec を役割 (roles) で列に束縛して解決。手書き index は不要:
        項の args シンボル名 (V / g) が列を決める。g を持つ項は選択 gate 上に展開
        (spec["gates"]=gate 序数リスト、既定=全 gate)。basis は spec["roles"] の
        役割名で列選択 (既定=全入力)。未知 type はエラー。"""
        from .catalog import FIXED_LIB_ENTRIES, VARIADIC_LIB_ENTRIES

        t = spec["type"]
        if t in VARIADIC_LIB_ENTRIES:
            cols = roles.basis_cols(spec.get("roles"), n_inputs)
            return [cls(entries=VARIADIC_LIB_ENTRIES[t](len(cols)), inputs=cols)]
        if t in FIXED_LIB_ENTRIES:
            entries = FIXED_LIB_ENTRIES[t]
            argnames = [s.name for s in entries[0].args]
            if "g" not in argnames:
                return [cls(entries=entries, inputs=roles.bind(argnames, None))]
            ords = spec.get("gates", range(len(roles.g)))
            return [
                cls(entries=entries, inputs=roles.bind(argnames, roles.g[k]))
                for k in ords
            ]
        known = sorted([*FIXED_LIB_ENTRIES, *VARIADIC_LIB_ENTRIES])
        raise ValueError(f"未知 library type: {t!r}。対応 type: {known}")

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
    def build(
        library_specs: list[dict], roles: "Roles", n_inputs: int
    ) -> "FeatureLibrary":
        subs = [
            sub
            for spec in library_specs
            for sub in SubLibrary.expand(spec, roles, n_inputs)
        ]
        return FeatureLibrary(
            sub_libraries=subs,
            library=ps.GeneralizedLibrary(
                [sl.to_ps_library() for sl in subs],
                inputs_per_library=[sl.inputs for sl in subs],
            ),
        )
