# Traub 1991 CA3 pyramidal cell
# ref: tmp/dataset_utils/traub/traub.{c,h}

from typing import NamedTuple

import jax.numpy as jnp

from ..core.network import CompartmentType
from ..core.opcost import OpCost
from .common import lin_exp_form

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
    # 表面積 [cm^2]。u_t (絶対電流 [μA]) を密度 [μA/cm^2] に変換する除数。
    # default 1.0 → 単一 comp 用 (u_t を密度スケールで渡す既存挙動維持)
    area: float = 1.0


def traub_dv(p: TraubParams, u_t, v, states):
    """Traub 物理 dV/dt。states=10次元 (hybrid では decode 済 latent)。"""
    M, S, N, C, A, H, R, B, Q, XI = [states[i] for i in range(10)]
    i_leak = p.g_leak * (v - p.V_LEAK)
    i_na = p.g_Na * M * M * H * (v - p.V_Na)
    i_ca = p.g_Ca * S * S * R * (v - p.V_Ca)
    i_kdr = p.g_K_DR * N * (v - p.V_K)
    i_ka = p.g_K_A * A * B * (v - p.V_K)
    i_kahp = p.g_K_AHP * Q * (v - p.V_K)
    i_kc = p.g_K_C * C * jnp.minimum(1.0, XI / 250.0) * (v - p.V_K)
    i_ion = i_leak + i_na + i_ca + i_kdr + i_ka + i_kahp + i_kc
    # u_t は絶対電流 [μA] スケール (coupling: g_axial · dV, ext: μA)。密度化に /area。
    return (-i_ion + u_t / p.area) / p.Cm


def calc_traub_channel(p: TraubParams, u_t, v, states):
    """Traub: (dv, dstates 10次元) を返す"""
    M, S, N, C, A, H, R, B, Q, XI = [states[i] for i in range(10)]
    dv = traub_dv(p, u_t, v, states)

    dM = _traub_dstate_v("M", v, M)
    dS = _traub_dstate_v("S", v, S)
    dN = _traub_dstate_v("N", v, N)
    dC = _traub_dstate_v("C", v, C)
    dA = _traub_dstate_v("A", v, A)
    dH = _traub_dstate_v("H", v, H)
    dR = _traub_dstate_v("R", v, R)
    dB = _traub_dstate_v("B", v, B)
    dQ = traub_alpha_q(XI) * (1.0 - Q) - traub_beta_q(XI) * Q
    i_ca = p.g_Ca * S * S * R * (v - p.V_Ca)
    dXI = -p.phi_area * i_ca - p.Beta * XI

    return dv, jnp.stack([dM, dS, dN, dC, dA, dH, dR, dB, dQ, dXI])


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


def _traub_dstate_v(name, v, x):
    a, b = _TRAUB_RATE_V[name]
    return a(v) * (1.0 - x) - b(v) * x


# レート関数の演算コスト。_traub_u の pm=1、lin_exp_form の (exp=1, pm=1, div=1) を
# 含む。jnp.where の分岐は両枝とも評価されるため両方を積算 (比較自体は無視)。
TRAUB_RATE_COST_MAP: dict[str, OpCost] = {
    "traub_alpha_m": OpCost(exp=1, div=2, pm=3, mul=1),
    "traub_beta_m": OpCost(exp=1, div=2, pm=3, mul=1),
    "traub_alpha_s": OpCost(exp=1, div=1, pm=3, mul=1),
    "traub_beta_s": OpCost(exp=1, div=2, pm=3, mul=1),
    "traub_alpha_n": OpCost(exp=1, div=2, pm=3, mul=1),
    "traub_beta_n": OpCost(exp=1, div=1, pm=2, mul=1),
    "traub_alpha_c": OpCost(exp=2, div=4, pm=6, mul=1),
    "traub_beta_c": OpCost(exp=3, div=5, pm=10, mul=2),  # alpha_c 再計算を含む
    "traub_alpha_a": OpCost(exp=1, div=2, pm=3, mul=1),
    "traub_beta_a": OpCost(exp=1, div=2, pm=3, mul=1),
    "traub_alpha_h": OpCost(exp=1, div=1, pm=2, mul=1),
    "traub_beta_h": OpCost(exp=1, div=2, pm=3),
    "traub_alpha_r": OpCost(exp=1, div=2, pm=2),
    "traub_beta_r": OpCost(exp=1, div=2, pm=3),  # alpha_r 再計算を含む
    "traub_alpha_b": OpCost(exp=1, div=1, pm=2, mul=1),
    "traub_beta_b": OpCost(exp=1, div=2, pm=3),
}


# 物理 dV/dt 演算コスト (traub_dv): 7イオン電流 + i_ion 総和 + dv/dt
TRAUB_DV_COST = (
    OpCost(pm=1, mul=1)  # i_leak
    + OpCost(pm=1, mul=4)  # i_na
    + OpCost(pm=1, mul=4)  # i_ca
    + OpCost(pm=1, mul=2)  # i_kdr
    + OpCost(pm=1, mul=3)  # i_ka
    + OpCost(pm=1, mul=2)  # i_kahp
    + OpCost(pm=1, mul=3, div=1)  # i_kc (min は無視)
    + OpCost(pm=6)  # i_ion 総和
    + OpCost(pm=1, div=2)  # dv = (-i_ion + u_t/area) / Cm
)


_TRAUB_DEFAULT_PARAMS = TraubParams()


TRAUB_TYPE = CompartmentType(
    name="traub",
    kernel=calc_traub_channel,
    param_cls=TraubParams,
    gate_names=TRAUB_STATE_NAMES,
    default_gate_inits=_traub_state_inits(_TRAUB_DEFAULT_PARAMS),
    v_init=TRAUB_V_INIT,
    opcost=OpCost(),  # TODO: 実測 or 積算
)
