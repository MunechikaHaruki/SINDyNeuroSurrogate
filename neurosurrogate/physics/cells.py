"""区画の式。状態の名前ごとに微分の式を持ち、kernel・初期値・演算コストはそこから組む。

レートの式は隣の hh.json と traub.json が持つ (TS の evaluation/traub/traub.ts も読む)。
Traub の per-comp 定数は traub.c と代数的に等価。
区画は HH・Traub・passive と、Traub19 の区画の定数とネット (NETS)。kernel は
physical_cell が組む。
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp

# --- 演算コスト ------------------------------------------------------------------


@dataclass(frozen=True)
class OpCost:
    exp: int = 0
    div: int = 0
    pm: int = 0
    mul: int = 0

    def __add__(self, other: "OpCost") -> "OpCost":
        return OpCost(
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in fields(self)
            }
        )

    def __mul__(self, n: int) -> "OpCost":
        return OpCost(**{f.name: getattr(self, f.name) * n for f in fields(self)})

    def to_dict(self) -> dict[str, int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def total(costs: list[OpCost]) -> OpCost:
    """コストの和。"""
    return sum(costs, OpCost())


# --- 区画の型 --------------------------------------------------------------------


@dataclass(frozen=True)
class CellType:
    """区画の型。kernel は (params, 電流密度, V, 状態) → (dV/dt, d状態/dt)。

    元のモデルは物理の式 (dv・derivs・costs) も持ち、サロゲートは学習しなかった状態を
    それで解く。状態の並びは states、初期値は inits(params) = [V, *states]。
    """

    name: str
    kernel: Callable
    param_cls: type | None
    states: list[str]
    inits: Callable[[Any], list[float]]
    opcost: OpCost
    dv: Callable | None = None  # (p, u, v, s: 名前 → 値) → dV/dt
    derivs: dict[str, Callable] | None = None  # 名前 → (p, v, s) → d/dt
    costs: dict[str, OpCost] | None = None  # 名前 → その状態を物理で解くコスト
    dv_cost: OpCost | None = None


def physical_cell(
    name: str,
    param_cls: type,
    states: list[str],
    dv: Callable,
    derivs: dict[str, Callable],
    inits: Callable[[Any], list[float]],
    costs: dict[str, OpCost],
    dv_cost: OpCost,
) -> CellType:
    """物理の式から区画の型を組む。kernel は全状態を名前で微分する。"""

    def kernel(p, u, v, x):
        s = {n: x[i] for i, n in enumerate(states)}
        return dv(p, u, v, s), jnp.stack([derivs[n](p, v, s) for n in states])

    return CellType(
        name=name,
        kernel=kernel,
        param_cls=param_cls,
        states=states,
        inits=inits,
        opcost=dv_cost + total([costs[n] for n in states]),
        dv=dv,
        derivs=derivs,
        costs=costs,
        dv_cost=dv_cost,
    )


# --- レートの形 ------------------------------------------------------------------


def _lin_exp_form(x):
    """x / (e^x − 1)。|x| < 1e-4 はテイラー展開 (TS の bernoulli と同じ)。"""
    small = jnp.abs(x) < 1e-4
    safe = jnp.where(small, 1.0, x)
    return jnp.where(
        small,
        1.0 - x / 2.0 + x**2 / 12.0 - x**4 / 720.0,
        safe / jnp.expm1(safe),
    )


_FORMS = {
    "exp": lambda a, z: a * jnp.exp(z),
    "sigmoid": lambda a, z: a / (1.0 + jnp.exp(z)),
    "linexp": lambda a, z: a * _lin_exp_form(z),
}
# 形の演算コスト (_lin_exp_form は exp=1, pm=1, div=1)。
_FORM_COSTS = {
    "exp": OpCost(exp=1, mul=1),
    "sigmoid": OpCost(exp=1, pm=1, div=1),
    "linexp": OpCost(exp=1, pm=1, div=1, mul=1),
}


def _form_rate(spec):
    """{form, A, theta, s} → u ↦ A·form((θ − u)/s)。"""
    form = _FORMS[spec["form"]]

    def rate(u):
        return form(spec["A"], (spec["theta"] - u) / spec["s"])

    return rate


def _form_cost(spec) -> OpCost:
    """(θ − u)/s の pm=1, div=1 と形の分。"""
    return OpCost(pm=1, div=1) + _FORM_COSTS[spec["form"]]


def _gate_ode(alpha, beta):
    def dxdt(v, x):
        return alpha(v) * (1.0 - x) - beta(v) * x

    return dxdt


def _inf(alpha, beta):
    def inf(v):
        return alpha(v) / (alpha(v) + beta(v))

    return inf


def _read(name: str) -> dict:
    spec: dict = json.loads(Path(__file__).with_name(f"{name}.json").read_text())
    return spec


_GATE_ODE_COST = OpCost(pm=2, mul=2)  # dg = α·(1 − g) − β·g

# レート関数。名前は SINDy の項の記号と、項の演算コストの鍵にもなる。
RATES: dict[str, Callable] = {}
RATE_COSTS: dict[str, OpCost] = {}


# --- HH --------------------------------------------------------------------------

_HH_RATES = _read("hh")["rates"]
for _gate, _pair in _HH_RATES.items():
    for _kind, _spec in _pair.items():
        RATES[f"{_kind}_{_gate}_hh"] = _form_rate(_spec)
        RATE_COSTS[f"{_kind}_{_gate}_hh"] = _form_cost(_spec)


class HHParams(NamedTuple):
    E_REST: float = -65.0
    C: float = 1.0
    G_LEAK: float = 0.3
    E_LEAK: float = 10.6 - 65.0
    G_NA: float = 120.0
    E_NA: float = 115.0 - 65.0
    G_K: float = 36.0
    E_K: float = -12.0 - 65.0


class PassiveParams(NamedTuple):
    C: float = 1.0
    G_LEAK: float = 0.3
    E_LEAK: float = 10.6 - 65.0


def _hh_dv(p: HHParams, u, v, s):
    i_na = p.G_NA * s["M"] ** 3 * s["H"] * (v - p.E_NA)
    i_k = p.G_K * s["N"] ** 4 * (v - p.E_K)
    i_leak = p.G_LEAK * (v - p.E_LEAK)
    return (-i_leak - i_na - i_k + u) / p.C


def _hh_gate(gate: str):
    """HH のゲートの微分。レートは静止電位からの電位で読む。"""
    dxdt = _gate_ode(RATES[f"alpha_{gate}_hh"], RATES[f"beta_{gate}_hh"])

    def deriv(p, v, s):
        return dxdt(v - p.E_REST, s[gate.upper()])

    return deriv


def _hh_inits(p: HHParams) -> list[float]:
    """V を自身の E_REST に置くので v_rel = 0 で定常。"""
    gates = []
    for gate in "mhn":
        gates.append(
            float(_inf(RATES[f"alpha_{gate}_hh"], RATES[f"beta_{gate}_hh"])(0.0))
        )
    return [p.E_REST, *gates]


def _rate_pair_cost(gate: str, model: str) -> OpCost:
    return RATE_COSTS[f"alpha_{gate}_{model}"] + RATE_COSTS[f"beta_{gate}_{model}"]


HH = physical_cell(
    name="hh",
    param_cls=HHParams,
    states=["M", "H", "N"],
    dv=_hh_dv,
    derivs={g.upper(): _hh_gate(g) for g in "mhn"},
    inits=_hh_inits,
    costs={g.upper(): _rate_pair_cost(g, "hh") + _GATE_ODE_COST for g in "mhn"},
    dv_cost=(
        OpCost(pm=1)  # 反転電位
        + OpCost(pm=3, mul=5) * 2  # Na, K 電流
        + OpCost(pm=1, mul=1)  # leak 電流
        + OpCost(pm=3, div=1)  # dv/dt
    ),
)


def _passive_kernel(p: PassiveParams, u, v, x):
    return (-p.G_LEAK * (v - p.E_LEAK) + u) / p.C, x


PASSIVE = CellType(
    name="passive",
    kernel=_passive_kernel,
    param_cls=PassiveParams,
    states=[],
    inits=lambda _: [-65.0],  # passive は静止電位を規定する params を持たない
    opcost=OpCost(div=1, pm=2, mul=1),
)


# --- Traub (1991 CA3 錐体細胞) ---------------------------------------------------

_TRAUB = _read("traub")
TRAUB_V_LEAK = _TRAUB["v_rest"]


def _traub_rate(spec):
    """traub.json の 1 本を、原著の基準電位 u = v − V_LEAK で読む。"""
    rate_u = _form_rate(spec)

    def rate(v):
        return rate_u(v - TRAUB_V_LEAK)

    return rate


for _gate, _pair in _TRAUB["rates"].items():
    for _kind, _spec in _pair.items():
        RATES[f"{_kind}_{_gate}_traub"] = _traub_rate(_spec)
        RATE_COSTS[f"{_kind}_{_gate}_traub"] = OpCost(pm=1) + _form_cost(_spec)


# c と r は電位で場合分けし、β を α との差で置くので三つの形に収まらない。
# TS と両方で手書きする。
def _alpha_c(v):
    u = v - TRAUB_V_LEAK
    low = jnp.exp((u - 10.0) / 11.0 - (u - 6.5) / 27.0) / 18.975
    high = 2.0 * jnp.exp(-(u - 6.5) / 27.0)
    return jnp.where(v <= 50.0 + TRAUB_V_LEAK, low, high)


def _beta_c(v):
    u = v - TRAUB_V_LEAK
    return jnp.where(
        v <= 50.0 + TRAUB_V_LEAK, 2.0 * jnp.exp(-(u - 6.5) / 27.0) - _alpha_c(v), 0.0
    )


def _alpha_r(v):
    u = v - TRAUB_V_LEAK
    return jnp.where(v <= TRAUB_V_LEAK, 0.005, jnp.exp(-u / 20.0) / 200.0)


def _beta_r(v):
    return jnp.where(v <= TRAUB_V_LEAK, 0.0, 0.005 - _alpha_r(v))


RATES |= {
    "alpha_c_traub": _alpha_c,
    "beta_c_traub": _beta_c,
    "alpha_r_traub": _alpha_r,
    "beta_r_traub": _beta_r,
}
# jnp.where の分岐は両枝とも評価されるので両方を積算する (比較自体は無視)。
RATE_COSTS |= {
    "alpha_c_traub": OpCost(exp=2, div=4, pm=6, mul=1),
    "beta_c_traub": OpCost(exp=3, div=5, pm=10, mul=2),  # alpha_c 再計算を含む
    "alpha_r_traub": OpCost(exp=1, div=2, pm=2),
    "beta_r_traub": OpCost(exp=1, div=2, pm=3),  # alpha_r 再計算を含む
}


def _alpha_q(xi):
    return jnp.minimum(0.2e-4 * xi, 0.01)


def _beta_q(xi):
    return jnp.full_like(xi, 0.001)


_dq = _gate_ode(_alpha_q, _beta_q)  # Ca 濃度 XI で駆動される AHP ゲート Q
_inf_q = _inf(_alpha_q, _beta_q)


class TraubParams(NamedTuple):
    Cm: float = 3.0
    Beta: float = 0.075
    V_LEAK: float = -60.0
    V_Na: float = 115.0 - 60.0
    V_Ca: float = 140.0 - 60.0
    V_K: float = -15.0 - 60.0
    # soma-like defaults (traub.c index 8 相当)
    g_leak: float = 0.1
    g_Na: float = 30.0
    g_Ca: float = 4.0
    g_K_DR: float = 15.0
    g_K_A: float = 5.0
    g_K_AHP: float = 0.8
    g_K_C: float = 10.0
    phi_area: float = 17402.0 * 3.320e-5  # phi * area
    # 表面積 [cm^2]。軸索 coupling (絶対量 [μA]) を密度にする除数。1.0 は単一区画用。
    area: float = 1.0


TRAUB_STATES = ["M", "N", "C", "A", "H", "B", "S", "R", "Q", "XI"]
_TRAUB_GATES = ["M", "N", "C", "A", "H", "B", "S", "R"]  # 電位だけで動くゲート


def _traub_dv(p: TraubParams, u, v, s):
    i_leak = p.g_leak * (v - p.V_LEAK)
    i_na = p.g_Na * s["M"] * s["M"] * s["H"] * (v - p.V_Na)
    i_ca = p.g_Ca * s["S"] * s["S"] * s["R"] * (v - p.V_Ca)
    i_kdr = p.g_K_DR * s["N"] * (v - p.V_K)
    i_ka = p.g_K_A * s["A"] * s["B"] * (v - p.V_K)
    i_kahp = p.g_K_AHP * s["Q"] * (v - p.V_K)
    i_kc = p.g_K_C * s["C"] * jnp.minimum(1.0, s["XI"] / 250.0) * (v - p.V_K)
    i_ion = i_leak + i_na + i_ca + i_kdr + i_ka + i_kahp + i_kc
    return (-i_ion + u) / p.Cm


def _traub_gate(gate: str):
    dxdt = _gate_ode(RATES[f"alpha_{gate}_traub"], RATES[f"beta_{gate}_traub"])

    def deriv(p, v, s):
        return dxdt(v, s[gate.upper()])

    return deriv


def _traub_dxi(p: TraubParams, v, s):
    i_ca = p.g_Ca * s["S"] * s["S"] * s["R"] * (v - p.V_Ca)
    return -p.phi_area * i_ca - p.Beta * s["XI"]


def _traub_dq(p: TraubParams, v, s):
    return _dq(s["XI"], s["Q"])


def _traub_inf(gate: str, v) -> float:
    return float(_inf(RATES[f"alpha_{gate}_traub"], RATES[f"beta_{gate}_traub"])(v))


def _traub_inits(p: TraubParams) -> list[float]:
    """静止電位 V_LEAK で定常。XI は dXI/dt = 0 となる濃度。"""
    v0 = p.V_LEAK
    gates = {g: _traub_inf(g.lower(), v0) for g in _TRAUB_GATES}
    i_ca0 = p.g_Ca * gates["S"] * gates["S"] * gates["R"] * (v0 - p.V_Ca)
    xi = float(-p.phi_area * i_ca0 / p.Beta)
    q = float(_inf_q(jnp.asarray(xi)))
    return [v0, *(gates[g] for g in _TRAUB_GATES), q, xi]


TRAUB = physical_cell(
    name="traub",
    param_cls=TraubParams,
    states=TRAUB_STATES,
    dv=_traub_dv,
    derivs={g: _traub_gate(g.lower()) for g in _TRAUB_GATES}
    | {"Q": _traub_dq, "XI": _traub_dxi},
    inits=_traub_inits,
    costs={
        g: _rate_pair_cost(g.lower(), "traub") + _GATE_ODE_COST for g in _TRAUB_GATES
    }
    | {
        "Q": OpCost(pm=2, mul=3),  # dQ = α_q·(1 − Q) − β_q·Q (min と定数は無視)
        "XI": OpCost(pm=1, mul=4) + OpCost(pm=1, mul=2),  # i_ca と dXI
    },
    dv_cost=(
        OpCost(pm=1, mul=1)  # i_leak
        + OpCost(pm=1, mul=4)  # i_na
        + OpCost(pm=1, mul=4)  # i_ca
        + OpCost(pm=1, mul=2)  # i_kdr
        + OpCost(pm=1, mul=3)  # i_ka
        + OpCost(pm=1, mul=2)  # i_kahp
        + OpCost(pm=1, mul=3, div=1)  # i_kc (min は無視)
        + OpCost(pm=6)  # i_ion の総和
        + OpCost(pm=1, div=1)  # dv = (−i_ion + u) / Cm
    ),
)

CELLS = {"hh": HH, "passive": PASSIVE, "traub": TRAUB}


# --- ネット (区画のつながり) -----------------------------------------------------


@dataclass(frozen=True)
class Compartment:
    """ネットの 1 ノード。型と、型の既定と違う params だけを持つ。"""

    name: str
    type: CellType
    params: tuple | None = None

    @property
    def resolved_params(self) -> tuple | None:
        """明示 params、無ければ型の既定。"""
        if self.params is not None:
            return self.params
        return self.type.param_cls() if self.type.param_cls is not None else None

    @property
    def init(self) -> list[float]:
        """[V, *states] の初期値。型 × 自身の params から毎回解く。"""
        return self.type.inits(self.resolved_params)

    @property
    def area(self) -> float:
        """膜面積 [cm^2]。kernel は電流を密度で受けるので、coupling をこれで割る。"""
        return getattr(self.resolved_params, "area", 1.0)


@dataclass
class Edge:
    src: str
    dst: str
    weight: float  # 軸方向 conductance [μS]


@dataclass
class Net:
    """形態だけ。どこへ電流を入れるかは実験の記述 (SimSpec.stim) が持つ。"""

    nodes: list[Compartment]
    edges: list[Edge]

    @cached_property
    def _index(self) -> dict[str, int]:
        return {c.name: i for i, c in enumerate(self.nodes)}

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.nodes]

    def index(self, name: str) -> int:
        return self._index[name]


# --- Traub 19 区画 (traub.c の per-comp 定数) ------------------------------------

_G_NA = [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 15.0, 30.0, 15.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # noqa: E501  # fmt: skip
_G_K_DR = [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 5.0, 15.0, 5.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # noqa: E501  # fmt: skip
_G_K_A = [0.0] * 8 + [5.0] + [0.0] * 10  # soma だけ 5.0
_G_K_C = [0.0, 5.0, 5.0, 10.0, 10.0, 10.0, 5.0, 20.0, 10.0, 20.0, 5.0, 15.0, 15.0, 15.0, 15.0, 15.0, 5.0, 5.0, 0.0]  # noqa: E501  # fmt: skip
_G_K_AHP = [0.0] + [0.8] * 17 + [0.0]
_G_CA = [0.0, 5.0, 5.0, 12.0, 12.0, 12.0, 5.0, 8.0, 4.0, 8.0, 5.0, 17.0, 17.0, 17.0, 10.0, 10.0, 5.0, 5.0, 0.0]  # noqa: E501  # fmt: skip
_PHI = [7769.0] * 7 + [34530.0, 17402.0, 26404.0] + [5941.0] * 9
_RAD = [2.89e-4] * 8 + [4.23e-4] + [2.42e-4] * 10
_LEN = [1.20e-2] * 8 + [1.25e-2] + [1.10e-2] * 10
_AREA = [2.188e-5] * 8 + [3.320e-5] + [1.673e-5] * 10
_RI = 0.1  # KΩ·cm
TRAUB19_SOMA = 8


def traub19_name(i: int) -> str:
    """細胞体は soma、他は c00..c18。"""
    return "soma" if i == TRAUB19_SOMA else f"c{i:02d}"


def traub19_params(i: int) -> TraubParams:
    return TraubParams(
        g_leak=0.1,
        g_Na=_G_NA[i],
        g_Ca=_G_CA[i],
        g_K_DR=_G_K_DR[i],
        g_K_A=_G_K_A[i],
        g_K_AHP=_G_K_AHP[i],
        g_K_C=_G_K_C[i],
        phi_area=_PHI[i] * _AREA[i],
        area=_AREA[i],
    )


def _g_axial(i: int) -> float:
    """区画 i と i+1 の間の軸方向 conductance [μS]。"""
    r_i = _RI * _LEN[i] / (math.pi * _RAD[i] ** 2)
    r_ip1 = _RI * _LEN[i + 1] / (math.pi * _RAD[i + 1] ** 2)
    return 2.0 / (r_i + r_ip1)


def _traub19() -> Net:
    nodes = [Compartment(traub19_name(i), TRAUB, traub19_params(i)) for i in range(19)]
    edges = [Edge(traub19_name(i), traub19_name(i + 1), _g_axial(i)) for i in range(18)]
    return Net(nodes, edges)


# SimSpec.target が名前で引く適用先。
NETS: dict[str, Net] = {
    "hh": Net([Compartment("soma", HH)], []),
    "traub": Net([Compartment("soma", TRAUB)], []),
    "traub19": _traub19(),
}
