from typing import NamedTuple

import jax.numpy as jnp

from ..core.network import CompartmentType
from ..core.opcost import OpCost
from ._common import form_rate, form_rate_cost, gate_ode, inf_ode, read_spec

# レートの式は隣の hh.json が持つ。
# 引数は静止電位からの電位 v_rel (calc_hh_channel が引く)。
_RATES = read_spec("hh")["rates"]

alpha_m_hh = form_rate(_RATES["m"]["alpha"])
beta_m_hh = form_rate(_RATES["m"]["beta"])
alpha_h_hh = form_rate(_RATES["h"]["alpha"])
beta_h_hh = form_rate(_RATES["h"]["beta"])
alpha_n_hh = form_rate(_RATES["n"]["alpha"])
beta_n_hh = form_rate(_RATES["n"]["beta"])


m_inf = inf_ode(alpha_m_hh, beta_m_hh)
h_inf = inf_ode(alpha_h_hh, beta_h_hh)
n_inf = inf_ode(alpha_n_hh, beta_n_hh)
dmdt = gate_ode(alpha_m_hh, beta_m_hh)
dhdt = gate_ode(alpha_h_hh, beta_h_hh)
dndt = gate_ode(alpha_n_hh, beta_n_hh)


# --- Params クラス (データのみ、NamedTuple) ---


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


# --- Kernel 関数 (物理式、モジュール関数) ---


def hh_dv(p: HHParams, u_t, v, gates):
    """HH 物理 dV/dt。gates=(m,h,n) はゲート値 (hybrid では decode 済 latent)。"""
    m, h, n = gates[0], gates[1], gates[2]
    i_na = p.G_NA * m**3 * h * (v - p.E_NA)
    i_k = p.G_K * n**4 * (v - p.E_K)
    i_leak = p.G_LEAK * (v - p.E_LEAK)
    return (-i_leak - i_na - i_k + u_t) / p.C


def calc_hh_channel(p: HHParams, u_t, v, curr_gate):
    """HH: (dv, dgate) を返す"""
    m, h, n = curr_gate[0], curr_gate[1], curr_gate[2]
    v_rel = v - p.E_REST
    dgate = jnp.stack([dmdt(v_rel, m), dhdt(v_rel, h), dndt(v_rel, n)])
    return hh_dv(p, u_t, v, curr_gate), dgate


def calc_passive_channel(p: PassiveParams, u_t, v, state):
    """Passive: (dv, state素通し)"""
    dv = (-p.G_LEAK * (v - p.E_LEAK) + u_t) / p.C
    return dv, state


# --- OpCost ---


# レート関数の演算コスト。hh.json の形から導く。
HH_RATE_COST_MAP: dict[str, OpCost] = {
    f"{kind}_{gate}_hh": form_rate_cost(spec)
    for gate, pair in _RATES.items()
    for kind, spec in pair.items()
}


HH_DV_COST = (
    OpCost(pm=1)  # 反転電位
    + OpCost(pm=3, mul=5) * 2  # Na,K電流
    + OpCost(pm=1, mul=1)  # leak電流
    + OpCost(pm=3, div=1)  # dv/dt
)

_HH_OPCOST = (
    sum(HH_RATE_COST_MAP.values(), OpCost())  # レート関数
    + HH_DV_COST
    + OpCost(pm=6, mul=6)  # dg/dt
)


PASSIVE_V_INIT = -65.0  # passive は静止電位を規定する params を持たない (E_LEAK≠静止)


# --- CompartmentType (物理の型) ---


def hh_inits(p: HHParams) -> list[float]:
    """[V, M, H, N] 初期状態。V を自身の E_REST に置くので v_rel=0 で定常。"""
    return [p.E_REST, float(m_inf(0.0)), float(h_inf(0.0)), float(n_inf(0.0))]


def passive_inits(_: PassiveParams) -> list[float]:
    return [PASSIVE_V_INIT]


HH_TYPE = CompartmentType(
    name="hh",
    kernel=calc_hh_channel,
    param_cls=HHParams,
    gate_names=["M", "H", "N"],
    inits=hh_inits,
    opcost=_HH_OPCOST,
)


PASSIVE_TYPE = CompartmentType(
    name="passive",
    kernel=calc_passive_channel,
    param_cls=PassiveParams,
    gate_names=[],
    inits=passive_inits,
    opcost=OpCost(div=1, pm=2, mul=1),
)
