from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pysindy as ps
import sympy as sp
from sympy.core.function import AppliedUndef

from ...compartments.hh import HH_RATE_COST_MAP
from ...compartments.traub import TRAUB_RATE_COST_MAP
from ...core.opcost import OpCost

_RATE_COST_MAP = HH_RATE_COST_MAP | TRAUB_RATE_COST_MAP

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
    和→pm(項数-1)、未定義レート関数→_RATE_COST_MAP、葉(記号/数)→0。"""
    match e:
        case _ if e.is_Symbol or e.is_Number:
            return OpCost()
        case AppliedUndef():
            return _RATE_COST_MAP[e.func.__name__]
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
    def arity(self) -> int:
        return len(self.args)

    @property
    def argnames(self) -> tuple[str, ...]:
        """args のシンボル名 (V/g/u)。この項が何の列を要求するかを表す = 列束縛の
        キーであり、展開規則 (g→latent 複製 / u→u 無し ansatz で脱落) の判定源。"""
        return tuple(s.name for s in self.args)

    def bound_expr(self, *cols: sp.Symbol) -> sp.Expr:
        """位置引数を列シンボルへ束縛した式。この式が feature の同一性そのもので、
        コストも (op_cost で) ここから出る。文字列化は表示と pysindy 境界のみ。"""
        return self.expr.subs(dict(zip(self.args, cols, strict=True)))

    def name_func(self, *names: str) -> str:
        """pysindy の function_names コールバック (str で呼ばれ str を返す境界)。"""
        return str(self.bound_expr(*map(sp.Symbol, names)))


def _group_by_argnames(
    entries: list[LibraryEntry],
) -> dict[tuple[str, ...], list[LibraryEntry]]:
    """argnames が同じ項をまとめる (挿入順保持)。argnames が同じ = 束縛先の列も
    arity も同じ → まとめて 1 束 (= SubLibrary 1 つ) に収まる。"""
    groups: defaultdict[tuple[str, ...], list[LibraryEntry]] = defaultdict(list)
    for e in entries:
        groups[e.argnames].append(e)
    return groups


@dataclass(frozen=True)
class SubLibrary:
    """1 library_spec を解決した 1 束。entries + 入力列 binding。"""

    entries: list[LibraryEntry]
    inputs: list[int]

    @classmethod
    def expand(cls, spec: dict, roles: "Roles") -> list["SubLibrary"]:
        """1 library_spec を役割 (roles) で列に束縛して解決。spec = {type, latents?}
        のみ: 手書き index も役割名も不要で、指定できる番号は latent (隠れ変数) の
        序数だけ (spec["latents"]、既定=全 latent)。展開の仕方は項の args シンボル名
        が決める: g を持つ項は選択 latent ごとに複製、持たない項は latent 非依存で
        1 束、u を持つ項は u の無い ansatz では落ちる。未知 type はエラー。"""
        from .catalog import LIB_ENTRIES

        if (t := spec["type"]) not in LIB_ENTRIES:
            raise ValueError(
                f"未知 library type: {t!r}。対応 type: {sorted(LIB_ENTRIES)}"
            )
        latents = spec.get("latents", range(len(roles.g)))
        subs: list[SubLibrary] = []
        for argnames, group in _group_by_argnames(LIB_ENTRIES[t]).items():
            if "u" in argnames and roles.u is None:
                continue
            if "g" in argnames:
                subs += [
                    cls(entries=group, inputs=roles.bind(argnames, roles.g[k]))
                    for k in latents
                ]
            else:
                subs.append(cls(entries=group, inputs=roles.bind(argnames, None)))
        return subs

    def to_ps_library(self) -> ps.CustomLibrary:
        return ps.CustomLibrary(
            library_functions=[e.func for e in self.entries],
            function_names=[e.name_func for e in self.entries],
        )

    def bound_exprs(self, columns: list[sp.Symbol]) -> list[sp.Expr]:
        """この束の項を実際の列シンボルへ束縛した式列 (feature 順)。"""
        bound = [columns[i] for i in self.inputs]
        return [e.bound_expr(*bound) for e in self.entries]


@dataclass(frozen=True)
class FeatureLibrary:
    sub_libraries: list[SubLibrary]
    library: ps.GeneralizedLibrary

    def bound_exprs(self, columns: list[sp.Symbol]) -> list[sp.Expr]:
        """全 feature の束縛式 (列順 = compute_theta = pysindy の feature 順)。
        式の重複 = 列の重複 → library type が互いに素という不変条件の破れ。"""
        exprs = [e for sl in self.sub_libraries for e in sl.bound_exprs(columns)]
        if dup := [e for e, n in Counter(exprs).items() if n > 1]:
            raise ValueError(f"library間で feature 重複: {sorted(map(str, dup))}")
        return exprs

    @staticmethod
    def build(library_specs: list[dict], roles: "Roles") -> "FeatureLibrary":
        subs = [sub for spec in library_specs for sub in SubLibrary.expand(spec, roles)]
        return FeatureLibrary(
            sub_libraries=subs,
            library=ps.GeneralizedLibrary(
                [sl.to_ps_library() for sl in subs],
                inputs_per_library=[sl.inputs for sl in subs],
            ),
        )
