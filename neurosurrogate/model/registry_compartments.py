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


# Traub 1991 CA3 pyramidal cell
# ref: tmp/dataset_utils/traub/traub.{c,h}

TRAUB_V_LEAK = -60.0
TRAUB_V_INIT = -60.0


def _traub_u(v):
    return v - TRAUB_V_LEAK


def traub_alpha_m(v):
    return 0.32 * 4.0 * lin_exp_form((13.1 - _traub_u(v)) / 4.0)


def traub_beta_m(v):
    return 0.28 * 5.0 * lin_exp_form((_traub_u(v) - 40.1) / 5.0)


def traub_alpha_s(v):
    return 1.6 / (1.0 + jnp.exp(-0.072 * (_traub_u(v) - 65.0)))


def traub_beta_s(v):
    return 0.02 * 5.0 * lin_exp_form((_traub_u(v) - 51.1) / 5.0)


def traub_alpha_n(v):
    return 0.016 * 5.0 * lin_exp_form((35.1 - _traub_u(v)) / 5.0)


def traub_beta_n(v):
    return 0.25 * jnp.exp((20.0 - _traub_u(v)) / 40.0)


def traub_alpha_c(v):
    u = _traub_u(v)
    low = jnp.exp((u - 10.0) / 11.0 - (u - 6.5) / 27.0) / 18.975
    high = 2.0 * jnp.exp(-(u - 6.5) / 27.0)
    return jnp.where(v <= 50.0 + TRAUB_V_LEAK, low, high)


def traub_beta_c(v):
    u = _traub_u(v)
    return jnp.where(
        v <= 50.0 + TRAUB_V_LEAK,
        2.0 * jnp.exp(-(u - 6.5) / 27.0) - traub_alpha_c(v),
        0.0,
    )


def traub_alpha_a(v):
    return 0.02 * 10.0 * lin_exp_form((13.1 - _traub_u(v)) / 10.0)


def traub_beta_a(v):
    return 0.0175 * 10.0 * lin_exp_form((_traub_u(v) - 40.1) / 10.0)


def traub_alpha_h(v):
    return 0.128 * jnp.exp((17.0 - _traub_u(v)) / 18.0)


def traub_beta_h(v):
    return 4.0 / (1.0 + jnp.exp((40.0 - _traub_u(v)) / 5.0))


def traub_alpha_r(v):
    u = _traub_u(v)
    return jnp.where(v <= TRAUB_V_LEAK, 0.005, jnp.exp(-u / 20.0) / 200.0)


def traub_beta_r(v):
    return jnp.where(v <= TRAUB_V_LEAK, 0.0, 0.005 - traub_alpha_r(v))


def traub_alpha_b(v):
    return 0.0016 * jnp.exp((-13.0 - _traub_u(v)) / 18.0)


def traub_beta_b(v):
    return 0.05 / (1.0 + jnp.exp((10.1 - _traub_u(v)) / 5.0))


def traub_alpha_q(xi):
    return jnp.minimum(0.2e-4 * xi, 0.01)


def traub_beta_q(xi):
    return jnp.full_like(xi, 0.001)


_TRAUB_RATE_V = {
    "M": (traub_alpha_m, traub_beta_m),
    "S": (traub_alpha_s, traub_beta_s),
    "N": (traub_alpha_n, traub_beta_n),
    "C": (traub_alpha_c, traub_beta_c),
    "A": (traub_alpha_a, traub_beta_a),
    "H": (traub_alpha_h, traub_beta_h),
    "R": (traub_alpha_r, traub_beta_r),
    "B": (traub_alpha_b, traub_beta_b),
}


def _traub_inf_v(name, v):
    a, b = _TRAUB_RATE_V[name]
    return a(v) / (a(v) + b(v))


def _traub_inf_q(xi):
    return traub_alpha_q(xi) / (traub_alpha_q(xi) + traub_beta_q(xi))


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


# 状態順序: [M, S, N, C, A, H, R, B, Q, XI]
TRAUB_STATE_NAMES = ["M", "S", "N", "C", "A", "H", "R", "B", "Q", "XI"]


def _traub_state_inits(p: TraubParams):
    v0 = p.V_LEAK
    m = float(_traub_inf_v("M", v0))
    s = float(_traub_inf_v("S", v0))
    n = float(_traub_inf_v("N", v0))
    c = float(_traub_inf_v("C", v0))
    a = float(_traub_inf_v("A", v0))
    h = float(_traub_inf_v("H", v0))
    r = float(_traub_inf_v("R", v0))
    b = float(_traub_inf_v("B", v0))
    # i_Ca(v0) を用いて XI 初期値を求める (定常: dXI/dt = 0)
    i_ca0 = p.g_Ca * s * s * r * (v0 - p.V_Ca)
    xi = float(-p.phi_area * i_ca0 / p.Beta)
    q = float(_traub_inf_q(jnp.asarray(xi)))
    return [m, s, n, c, a, h, r, b, q, xi]


def calc_traub_ion_currents(p: TraubParams, v, states):
    M, S, N, C, A, H, R, B, Q, XI = [states[i] for i in range(10)]
    i_leak = p.g_leak * (v - p.V_LEAK)
    i_na = p.g_Na * M * M * H * (v - p.V_Na)
    i_ca = p.g_Ca * S * S * R * (v - p.V_Ca)
    i_kdr = p.g_K_DR * N * (v - p.V_K)
    i_ka = p.g_K_A * A * B * (v - p.V_K)
    i_kahp = p.g_K_AHP * Q * (v - p.V_K)
    i_kc = p.g_K_C * C * jnp.minimum(1.0, XI / 250.0) * (v - p.V_K)
    return i_leak + i_na + i_ca + i_kdr + i_ka + i_kahp + i_kc, i_ca


def _traub_dstate_v(name, v, x):
    a, b = _TRAUB_RATE_V[name]
    return a(v) * (1.0 - x) - b(v) * x


def calc_traub_channel(p: TraubParams, u_t, v, states):
    i_ion, i_ca = calc_traub_ion_currents(p, v, states)
    dv = (-i_ion + u_t) / p.Cm

    M, S, N, C, A, H, R, B, Q, XI = [states[i] for i in range(10)]
    dM = _traub_dstate_v("M", v, M)
    dS = _traub_dstate_v("S", v, S)
    dN = _traub_dstate_v("N", v, N)
    dC = _traub_dstate_v("C", v, C)
    dA = _traub_dstate_v("A", v, A)
    dH = _traub_dstate_v("H", v, H)
    dR = _traub_dstate_v("R", v, R)
    dB = _traub_dstate_v("B", v, B)
    dQ = traub_alpha_q(XI) * (1.0 - Q) - traub_beta_q(XI) * Q
    dXI = -p.phi_area * i_ca - p.Beta * XI

    return dv, jnp.stack([dM, dS, dN, dC, dA, dH, dR, dB, dQ, dXI])


_TRAUB_DEFAULT_PARAMS = TraubParams()


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
    "traub": Compartment(
        type_name="traub",
        gate_inits=_traub_state_inits(_TRAUB_DEFAULT_PARAMS),
        gate_names=TRAUB_STATE_NAMES,
        v_init=TRAUB_V_INIT,
        OpCost=OpCost(),  # TODO: 実測 or 積算
    ),
}
