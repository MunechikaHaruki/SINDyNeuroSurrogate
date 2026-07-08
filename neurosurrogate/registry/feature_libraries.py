from collections.abc import Callable

from neurosurrogate.opcost import OpCost
from neurosurrogate.registry.compartments.hh import (
    HH_RATE_COST_MAP,
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
)
from neurosurrogate.surrogate.libraries import LibraryEntry

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


def _build_gate(func_names: list[str], is_product: bool) -> list[LibraryEntry]:
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


def _build_relaxation(is_decay: bool) -> list[LibraryEntry]:
    entries: list[LibraryEntry] = []
    for g in ["m", "h", "n"]:
        alpha_f, beta_f = GATE_PAIR_REGISTRY[g]
        a_name, b_name = alpha_f.__name__, beta_f.__name__
        if not is_decay:
            entries.append(
                LibraryEntry(
                    func=(lambda af: lambda x: af(x))(alpha_f),
                    name_func=(lambda an: lambda x: f"{an}({x})")(a_name),
                    cost=HH_RATE_COST_MAP[a_name],
                )
            )
        else:
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


GATE_ALL = ["alpha_m", "beta_m", "alpha_h", "beta_h", "alpha_n", "beta_n"]
GATE_FORWARD = ["alpha_m", "alpha_h", "alpha_n"]

# (type, arity) → LibraryEntry list。import 時 1 回計算 → 以降は定数辞書 lookup。
LIB_ENTRIES: dict[tuple[str, int], list[LibraryEntry]] = {
    ("gate", 1): _build_gate(GATE_ALL, is_product=False),
    ("gate", 2): _build_gate(GATE_ALL, is_product=True),
    ("gate_forward", 1): _build_gate(GATE_FORWARD, is_product=False),
    ("gate_forward", 2): _build_gate(GATE_FORWARD, is_product=True),
    ("relaxation", 1): _build_relaxation(is_decay=False),
    ("relaxation", 2): _build_relaxation(is_decay=True),
    ("volt", 3): _build_volt(),
    ("gate_poly_volt", 2): _build_gate_poly_volt(),
    ("basis", 2): _build_basis(2),
    ("basis", 3): _build_basis(3),
}
