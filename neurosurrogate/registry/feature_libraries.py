from collections.abc import Callable

from neurosurrogate.core.libraries import LibraryEntry
from neurosurrogate.metrics.opcost import OpCost
from neurosurrogate.registry.compartments.hh import (
    HH_GATE_FORWARD,
    HH_GATE_PAIRS,
    HH_RATE_ENTRIES,
)


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


_VOLT: list[LibraryEntry] = [
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


_GATE_POLY_VOLT: list[LibraryEntry] = []
for _p in range(1, 5):
    # g^p
    _GATE_POLY_VOLT.append(
        LibraryEntry(
            func=(lambda pp: lambda V, g: g**pp)(_p),
            name_func=(lambda pp: lambda V, g: f"{g}**{pp}")(_p),
            cost=OpCost(mul=max(0, _p - 1)),
        )
    )
    # g^p * V
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


# 固定 arity: type 名 → LibraryEntry list (import 時 1 回計算)
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

# 可変 arity: type 名 → arity 受けて LibraryEntry list を返す factory
VARIADIC_LIB_ENTRIES: dict[str, Callable[[int], list[LibraryEntry]]] = {
    "basis": _build_basis,
}
