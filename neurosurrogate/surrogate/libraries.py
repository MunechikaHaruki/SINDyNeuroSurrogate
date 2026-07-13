from collections.abc import Callable
from dataclasses import dataclass

import pysindy as ps

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


@dataclass(frozen=True)
class LibraryEntry:
    func: Callable
    name_func: Callable
    cost: OpCost

    def to_cost_entry(self, input_names: list[str]) -> tuple[str, OpCost]:
        return self.name_func(*input_names), self.cost


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
                expected = entries[0].func.__code__.co_argcount
                if len(inputs) != expected:
                    raise ValueError(
                        f"type={t!r} は arity={expected} 要求、"
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
# LibraryEntry カタログ (旧 registry/feature_libraries.py + hh.py の HH_* 部)
# ---------------------------------------------------------------------------


def _rate_entry(name: str, f, cost: OpCost) -> LibraryEntry:
    """レート関数 f を f(x) 形式の 1入力 LibraryEntry に。"""
    return LibraryEntry(
        func=f,
        name_func=(lambda nm: lambda x: f"{nm}({x})")(name),
        cost=cost,
    )


ALPHA_M_ENTRY = _rate_entry("alpha_m", alpha_m, HH_RATE_COST_MAP["alpha_m"])
BETA_M_ENTRY = _rate_entry("beta_m", beta_m, HH_RATE_COST_MAP["beta_m"])
ALPHA_H_ENTRY = _rate_entry("alpha_h", alpha_h, HH_RATE_COST_MAP["alpha_h"])
BETA_H_ENTRY = _rate_entry("beta_h", beta_h, HH_RATE_COST_MAP["beta_h"])
ALPHA_N_ENTRY = _rate_entry("alpha_n", alpha_n, HH_RATE_COST_MAP["alpha_n"])
BETA_N_ENTRY = _rate_entry("beta_n", beta_n, HH_RATE_COST_MAP["beta_n"])

HH_RATE_ENTRIES: list[LibraryEntry] = [
    ALPHA_M_ENTRY,
    BETA_M_ENTRY,
    ALPHA_H_ENTRY,
    BETA_H_ENTRY,
    ALPHA_N_ENTRY,
    BETA_N_ENTRY,
]

HH_GATE_PAIRS: list[tuple[LibraryEntry, LibraryEntry]] = [
    (ALPHA_M_ENTRY, BETA_M_ENTRY),
    (ALPHA_H_ENTRY, BETA_H_ENTRY),
    (ALPHA_N_ENTRY, BETA_N_ENTRY),
]

HH_GATE_FORWARD: list[LibraryEntry] = [ALPHA_M_ENTRY, ALPHA_H_ENTRY, ALPHA_N_ENTRY]


def _to_product(entry: LibraryEntry) -> LibraryEntry:
    """1入力 entry を (V, g) → f(V) * g の 2入力 entry に昇格。cost に mul 1 追加。"""
    base_f, base_name = entry.func, entry.name_func
    return LibraryEntry(
        func=(lambda fn: lambda x, y: fn(x) * y)(base_f),
        name_func=(lambda nfn: lambda x, y: f"{nfn(x)}*{y}")(base_name),
        cost=entry.cost + OpCost(mul=1),
    )


def _to_relaxation_decay(alpha_e: LibraryEntry, beta_e: LibraryEntry) -> LibraryEntry:
    """(alpha_entry, beta_entry) → (alpha(V) + beta(V)) * g の 2入力 entry。"""
    af, bf = alpha_e.func, beta_e.func
    an, bn = alpha_e.name_func, beta_e.name_func
    return LibraryEntry(
        func=(lambda af_, bf_: lambda x, y: (af_(x) + bf_(x)) * y)(af, bf),
        name_func=(lambda an_, bn_: lambda x, y: f"({an_(x)}+{bn_(x)})*{y}")(an, bn),
        cost=alpha_e.cost + beta_e.cost + OpCost(pm=1, mul=1),
    )


_VOLT: list[LibraryEntry] = [
    LibraryEntry(
        func=lambda u, v, w: u**3 * v * w,
        name_func=lambda u, v, w: f"{u}**3 * {v} * {w}",
        cost=OpCost(mul=4),
    ),
    LibraryEntry(
        func=lambda u, v, w: u**3 * v,
        name_func=lambda u, v, w: f"{u}**3 * {v}",
        cost=OpCost(mul=3),
    ),
    LibraryEntry(
        func=lambda u, v, w: u**4 * v,
        name_func=lambda u, v, w: f"{u}**4 * {v}",
        cost=OpCost(mul=4),
    ),
    LibraryEntry(
        func=lambda u, v, w: u**4,
        name_func=lambda u, v, w: f"{u}**4",
        cost=OpCost(mul=3),
    ),
]


_GATE_POLY_VOLT: list[LibraryEntry] = []
for _p in range(1, 5):
    _GATE_POLY_VOLT.append(
        LibraryEntry(
            func=(lambda pp: lambda V, g: g**pp)(_p),
            name_func=(lambda pp: lambda V, g: f"{g}**{pp}")(_p),
            cost=OpCost(mul=max(0, _p - 1)),
        )
    )
    _GATE_POLY_VOLT.append(
        LibraryEntry(
            func=(lambda pp: lambda V, g: g**pp * V)(_p),
            name_func=(lambda pp: lambda V, g: f"{g}**{pp} * {V}")(_p),
            cost=OpCost(mul=max(0, _p - 1) + 1),
        )
    )


def _make_projector(i: int, n: int) -> Callable:
    """n 引数関数で i 番目引数返す (co_argcount=n 保証)。"""
    args = ", ".join(f"x{j}" for j in range(n))
    src = f"def _p({args}): return x{i}\n"
    ns: dict = {}
    exec(src, ns)
    return ns["_p"]  # type: ignore[no-any-return]


def _make_constant(n: int) -> Callable:
    """n 引数関数で常に 1 返す。"""
    args = ", ".join(f"x{j}" for j in range(n))
    src = f"def _c({args}): return 1\n"
    ns: dict = {}
    exec(src, ns)
    return ns["_c"]  # type: ignore[no-any-return]


def _build_basis(n: int) -> list[LibraryEntry]:
    entries: list[LibraryEntry] = []
    for i in range(n):
        entries.append(
            LibraryEntry(
                func=_make_projector(i, n),
                name_func=(lambda ii: lambda *names: f"{names[ii]}")(i),
                cost=OpCost(),
            )
        )
    entries.append(
        LibraryEntry(
            func=_make_constant(n),
            name_func=lambda *names: "1",
            cost=OpCost(),
        )
    )
    return entries


FIXED_LIB_ENTRIES: dict[str, list[LibraryEntry]] = {
    "hh_gate": HH_RATE_ENTRIES,
    "hh_gate_product": [_to_product(e) for e in HH_RATE_ENTRIES],
    "hh_gate_forward": HH_GATE_FORWARD,
    "hh_gate_forward_product": [_to_product(e) for e in HH_GATE_FORWARD],
    "hh_relaxation_driver": [a for a, _ in HH_GATE_PAIRS],
    "hh_relaxation_decay": [_to_relaxation_decay(a, b) for a, b in HH_GATE_PAIRS],
    "volt": _VOLT,
    "gate_poly_volt": _GATE_POLY_VOLT,
}

VARIADIC_LIB_ENTRIES: dict[str, Callable[[int], list[LibraryEntry]]] = {
    "basis": _build_basis,
}
