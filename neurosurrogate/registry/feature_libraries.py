from neurosurrogate.features.libraries import LibraryEntry, SubLibrary
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


def gate(spec: dict) -> SubLibrary:
    """Gate 単体、または Gate * y のペア。"""
    func_names = spec["funcs"]
    is_product = spec.get("is_product", False)
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
    return SubLibrary(entries=entries, inputs=spec["inputs"])


def relaxation1(spec: dict) -> SubLibrary:
    """緩和ダイナミクスの駆動項: alpha_k(V)。inputs=[0] (V のみ)。"""
    entries: list[LibraryEntry] = []
    for g in spec["gates"]:
        alpha_f, _ = GATE_PAIR_REGISTRY[g]
        a_name = alpha_f.__name__
        entries.append(
            LibraryEntry(
                func=(lambda af: lambda x: af(x))(alpha_f),
                name_func=(lambda an: lambda x: f"{an}({x})")(a_name),
                cost=HH_RATE_COST_MAP[a_name],
            )
        )
    return SubLibrary(entries=entries, inputs=spec["inputs"])


def relaxation2(spec: dict) -> SubLibrary:
    """緩和ダイナミクスの減衰項: (alpha_k + beta_k)(V) * g0。inputs=[0, 1] (V, g0)。"""
    entries: list[LibraryEntry] = []
    for g in spec["gates"]:
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
    return SubLibrary(entries=entries, inputs=spec["inputs"])


def volt(spec: dict) -> SubLibrary:
    """V^a * g0^b * (...) 系の多項式項 (legacy)。"""
    entries = [
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
    return SubLibrary(entries=entries, inputs=spec["inputs"])


def gate_poly_volt(spec: dict) -> SubLibrary:
    """g^k と g^k * V (k=1..max_power) のペア。inputs 順は (V, g) で固定。"""
    max_power = spec["max_power"]
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
    return SubLibrary(entries=entries, inputs=spec["inputs"])


def identity(spec: dict) -> SubLibrary:
    """variable をそのまま渡す。zero cost。1 spec = 1 入力。"""
    entries = [
        LibraryEntry(
            func=lambda x: x,
            name_func=lambda x: f"{x}",
            cost=OpCost(),
        )
    ]
    return SubLibrary(entries=entries, inputs=spec["inputs"])


def const(spec: dict) -> SubLibrary:
    """定数項 1。zero cost。"""
    entries = [
        LibraryEntry(
            func=lambda: 1,
            name_func=lambda: "1",
            cost=OpCost(),
        )
    ]
    return SubLibrary(entries=entries, inputs=spec["inputs"])
