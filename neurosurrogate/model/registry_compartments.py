from typing import NamedTuple

import jax.numpy as jnp

from ..profiler.profiler_model import OpCost
from .model_dataset import Compartment


def lin_exp_form(x):
    denom = jnp.exp(x) - 1.0
    return jnp.where(
        jnp.abs(x) < 1e-8,
        1.0 / (1.0 + x / 2.0 + x**2 / 6.0 + x**3 / 24.0),
        x / jnp.where(denom == 0, 1.0, denom),
    )


def _inf_ode(alpha, beta):
    def inf(v):
        return alpha(v) / (alpha(v) + beta(v))

    return inf


def _gate_ode(alpha, beta):
    def dxdt(v, x):
        return alpha(v) * (1.0 - x) - beta(v) * x

    return dxdt


# rate functions


def alpha_m(v):
    return lin_exp_form(2.5 - 0.1 * v)


def beta_m(v):
    return 4.0 * jnp.exp(-v / 18.0)


def alpha_h(v):
    return 0.07 * jnp.exp(-v / 20.0)


def beta_h(v):
    return 1.0 / (jnp.exp(3.0 - 0.1 * v) + 1.0)


def alpha_n(v):
    return 0.1 * lin_exp_form(1 - 0.1 * v)


def beta_n(v):
    return 0.125 * jnp.exp(-v / 80.0)


# inf
m_inf = _inf_ode(alpha_m, beta_m)
h_inf = _inf_ode(alpha_h, beta_h)
n_inf = _inf_ode(alpha_n, beta_n)
# dgdt
dmdt = _gate_ode(alpha_m, beta_m)
dhdt = _gate_ode(alpha_h, beta_h)
dndt = _gate_ode(alpha_n, beta_n)


def calc_ion_currents(v, curr_gate, p):
    m, h, n = curr_gate[0], curr_gate[1], curr_gate[2]
    i_na = p.G_NA * m**3 * h * (v - p.E_NA)
    i_k = p.G_K * n**4 * (v - p.E_K)
    return i_na + i_k


def calc_i_leak(v, p):
    return p.G_LEAK * (v - p.E_LEAK)


def update_dvar_gate(v, gates):
    return jnp.stack([dmdt(v, gates[0]), dhdt(v, gates[1]), dndt(v, gates[2])])


def calc_hh_channel(p, u_t, v, curr_gate):
    return (
        (-calc_i_leak(v, p) - calc_ion_currents(v, curr_gate, p) + u_t) / p.C,
        update_dvar_gate(v - p.E_REST, curr_gate),
    )


def calc_passive_channel(p, u_t, v):
    return (-calc_i_leak(v, p) + u_t) / p.C


HH_RATE_COST_MAP: dict[str, OpCost] = {
    "alpha_m": OpCost(exp=1, div=1, pm=2, mul=2),
    "beta_m": OpCost(exp=1, div=1, pm=1, mul=1),
    "alpha_h": OpCost(exp=1, div=1, pm=1, mul=1),
    "beta_h": OpCost(exp=1, div=1, pm=2, mul=1),
    "alpha_n": OpCost(exp=1, div=1, pm=2, mul=2),
    "beta_n": OpCost(exp=1, div=1, pm=1, mul=1),
}


class HHParams(NamedTuple):
    E_REST: float = -65.0
    C: float = 1.0
    G_LEAK: float = 0.3
    E_LEAK: float = 10.6 - 65.0
    G_NA: float = 120.0
    E_NA: float = 115.0 - 65.0
    G_K: float = 36.0
    E_K: float = -12.0 - 65.0


V_REL = (-65) - (-65)  # V_INIT - E_REST


COMPARTMENT_TEMPLATES = {
    "hh": Compartment(
        type_name="hh",
        gate_inits=[float(m_inf(V_REL)), float(h_inf(V_REL)), float(n_inf(V_REL))],
        gate_names=["M", "H", "N"],
        OpCost=(
            sum(HH_RATE_COST_MAP.values(), OpCost())  # レート関数
            + OpCost(pm=1)  # 反転電位
            + OpCost(pm=3, mul=5) * 2  # Na,K電流
            + OpCost(pm=1, mul=1)  # leak電流
            + OpCost(pm=6, mul=6)  # dg/dt
            + OpCost(pm=3, div=1)  # dv/dtの計算
        ),
    ),
    "passive": Compartment(
        type_name="passive",
        gate_inits=[],
        gate_names=[],
        OpCost=OpCost(div=1, pm=2, mul=1),
    ),
}
