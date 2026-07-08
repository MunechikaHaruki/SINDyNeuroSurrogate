from collections.abc import Callable
from typing import TypeAlias

from neurosurrogate.opcost import OpCost
from neurosurrogate.registry.compartments.hh import (
    ALPHA_H_ENTRY,
    ALPHA_M_ENTRY,
    ALPHA_N_ENTRY,
    HH_GATE_PAIRS,
    HH_RATE_ENTRIES,
)
from neurosurrogate.surrogate.libraries import LibraryEntry

GatePair: TypeAlias = tuple[LibraryEntry, LibraryEntry]  # (alpha_entry, beta_entry)


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
    # alpha_k(V) + beta_k(V) → pm 1, それを g0 と乗算 → mul 1
    return LibraryEntry(
        func=(lambda af_, bf_: lambda x, y: (af_(x) + bf_(x)) * y)(af, bf),
        name_func=(lambda an_, bn_: lambda x, y: f"({an_(x)}+{bn_(x)})*{y}")(an, bn),
        cost=alpha_e.cost + beta_e.cost + OpCost(pm=1, mul=1),
    )


def _build_volt() -> list[LibraryEntry]:
    return [
        LibraryEntry(
            func=lambda u, v, w: u**3 * v * w,
            name_func=lambda u, v, w: f"{u}**3 * {v} * {w}",
            # u^3 = mul 2, *v = mul 1, *w = mul 1
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


def _build_gate_poly_volt() -> list[LibraryEntry]:
    max_power = 4
    entries: list[LibraryEntry] = []
    for p in range(1, max_power + 1):
        # g^p
        entries.append(
            LibraryEntry(
                func=(lambda pp: lambda V, g: g**pp)(p),
                name_func=(lambda pp: lambda V, g: f"{g}**{pp}")(p),
                cost=OpCost(mul=max(0, p - 1)),
            )
        )
        # g^p * V
        entries.append(
            LibraryEntry(
                func=(lambda pp: lambda V, g: g**pp * V)(p),
                name_func=(lambda pp: lambda V, g: f"{g}**{pp} * {V}")(p),
                cost=OpCost(mul=max(0, p - 1) + 1),
            )
        )
    return entries


def _make_projector(i: int, n: int) -> Callable:
    """n 引数関数で i 番目引数返す (co_argcount=n 保証)。"""
    args = ", ".join(f"x{j}" for j in range(n))
    src = f"def _p({args}): return x{i}\n"
    ns: dict = {}
    exec(src, ns)
    return ns["_p"]


def _make_constant(n: int) -> Callable:
    """n 引数関数で常に 1 返す。"""
    args = ", ".join(f"x{j}" for j in range(n))
    src = f"def _c({args}): return 1\n"
    ns: dict = {}
    exec(src, ns)
    return ns["_c"]


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


GATE_ALL: list[LibraryEntry] = HH_RATE_ENTRIES
GATE_FORWARD: list[LibraryEntry] = [ALPHA_M_ENTRY, ALPHA_H_ENTRY, ALPHA_N_ENTRY]
GATE_PAIRS: list[GatePair] = HH_GATE_PAIRS

# (type, arity) → LibraryEntry list。import 時 1 回計算 → 以降は定数辞書 lookup。
LIB_ENTRIES: dict[tuple[str, int], list[LibraryEntry]] = {
    ("gate", 1): GATE_ALL,
    ("gate", 2): [_to_product(e) for e in GATE_ALL],
    ("gate_forward", 1): GATE_FORWARD,
    ("gate_forward", 2): [_to_product(e) for e in GATE_FORWARD],
    ("relaxation", 1): [a for a, _ in GATE_PAIRS],
    ("relaxation", 2): [_to_relaxation_decay(a, b) for a, b in GATE_PAIRS],
    ("volt", 3): _build_volt(),
    ("gate_poly_volt", 2): _build_gate_poly_volt(),
    ("basis", 2): _build_basis(2),
    ("basis", 3): _build_basis(3),
}
