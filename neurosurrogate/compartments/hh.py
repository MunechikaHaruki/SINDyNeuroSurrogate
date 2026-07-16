from typing import NamedTuple

import jax.numpy as jnp

from ..core.network import CompartmentType
from ..core.opcost import OpCost
from .common import _gate_ode, _inf_ode, lin_exp_form


def alpha_m__hh(v):
    return lin_exp_form(2.5 - 0.1 * v)


def beta_m__hh(v):
    return 4.0 * jnp.exp(-v / 18.0)


def alpha_h__hh(v):
    return 0.07 * jnp.exp(-v / 20.0)


def beta_h__hh(v):
    return 1.0 / (jnp.exp(3.0 - 0.1 * v) + 1.0)


def alpha_n__hh(v):
    return 0.1 * lin_exp_form(1 - 0.1 * v)


def beta_n__hh(v):
    return 0.125 * jnp.exp(-v / 80.0)


m_inf = _inf_ode(alpha_m__hh, beta_m__hh)
h_inf = _inf_ode(alpha_h__hh, beta_h__hh)
n_inf = _inf_ode(alpha_n__hh, beta_n__hh)
dmdt = _gate_ode(alpha_m__hh, beta_m__hh)
dhdt = _gate_ode(alpha_h__hh, beta_h__hh)
dndt = _gate_ode(alpha_n__hh, beta_n__hh)


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


HH_RATE_COST_MAP: dict[str, OpCost] = {
    "alpha_m__hh": OpCost(exp=1, div=1, pm=2, mul=2),
    "beta_m__hh": OpCost(exp=1, div=1, pm=1, mul=1),
    "alpha_h__hh": OpCost(exp=1, div=1, pm=1, mul=1),
    "beta_h__hh": OpCost(exp=1, div=1, pm=2, mul=1),
    "alpha_n__hh": OpCost(exp=1, div=1, pm=2, mul=2),
    "beta_n__hh": OpCost(exp=1, div=1, pm=1, mul=1),
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


V_REL = (-65) - (-65)  # V_INIT - E_REST


# --- CompartmentType (物理の型) ---


HH_TYPE = CompartmentType(
    name="hh",
    kernel=calc_hh_channel,
    param_cls=HHParams,
    gate_names=["M", "H", "N"],
    default_gate_inits=[float(m_inf(V_REL)), float(h_inf(V_REL)), float(n_inf(V_REL))],
    v_init=-65,
    opcost=_HH_OPCOST,
)


PASSIVE_TYPE = CompartmentType(
    name="passive",
    kernel=calc_passive_channel,
    param_cls=PassiveParams,
    gate_names=[],
    default_gate_inits=[],
    v_init=-65,
    opcost=OpCost(div=1, pm=2, mul=1),
)
