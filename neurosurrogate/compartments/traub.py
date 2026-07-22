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


def alpha_m_traub(v):
    return 0.32 * 4.0 * lin_exp_form((13.1 - _traub_u(v)) / 4.0)


def beta_m_traub(v):
    return 0.28 * 5.0 * lin_exp_form((_traub_u(v) - 40.1) / 5.0)


def alpha_s_traub(v):
    return 1.6 / (1.0 + jnp.exp(-0.072 * (_traub_u(v) - 65.0)))


def beta_s_traub(v):
    return 0.02 * 5.0 * lin_exp_form((_traub_u(v) - 51.1) / 5.0)


def alpha_n_traub(v):
    return 0.016 * 5.0 * lin_exp_form((35.1 - _traub_u(v)) / 5.0)


def beta_n_traub(v):
    return 0.25 * jnp.exp((20.0 - _traub_u(v)) / 40.0)


def alpha_c_traub(v):
    u = _traub_u(v)
    low = jnp.exp((u - 10.0) / 11.0 - (u - 6.5) / 27.0) / 18.975
    high = 2.0 * jnp.exp(-(u - 6.5) / 27.0)
    return jnp.where(v <= 50.0 + TRAUB_V_LEAK, low, high)


def beta_c_traub(v):
    u = _traub_u(v)
    return jnp.where(
        v <= 50.0 + TRAUB_V_LEAK,
        2.0 * jnp.exp(-(u - 6.5) / 27.0) - alpha_c_traub(v),
        0.0,
    )


def alpha_a_traub(v):
    return 0.02 * 10.0 * lin_exp_form((13.1 - _traub_u(v)) / 10.0)


def beta_a_traub(v):
    return 0.0175 * 10.0 * lin_exp_form((_traub_u(v) - 40.1) / 10.0)


def alpha_h_traub(v):
    return 0.128 * jnp.exp((17.0 - _traub_u(v)) / 18.0)


def beta_h_traub(v):
    return 4.0 / (1.0 + jnp.exp((40.0 - _traub_u(v)) / 5.0))


def alpha_r_traub(v):
    u = _traub_u(v)
    return jnp.where(v <= TRAUB_V_LEAK, 0.005, jnp.exp(-u / 20.0) / 200.0)


def beta_r_traub(v):
    return jnp.where(v <= TRAUB_V_LEAK, 0.0, 0.005 - alpha_r_traub(v))


def alpha_b_traub(v):
    return 0.0016 * jnp.exp((-13.0 - _traub_u(v)) / 18.0)


def beta_b_traub(v):
    return 0.05 / (1.0 + jnp.exp((10.1 - _traub_u(v)) / 5.0))


def alpha_q_traub(xi):
    return jnp.minimum(0.2e-4 * xi, 0.01)


def beta_q_traub(xi):
    return jnp.full_like(xi, 0.001)


_TRAUB_RATE_V = {
    "M": (alpha_m_traub, beta_m_traub),
    "S": (alpha_s_traub, beta_s_traub),
    "N": (alpha_n_traub, beta_n_traub),
    "C": (alpha_c_traub, beta_c_traub),
    "A": (alpha_a_traub, beta_a_traub),
    "H": (alpha_h_traub, beta_h_traub),
    "R": (alpha_r_traub, beta_r_traub),
    "B": (alpha_b_traub, beta_b_traub),
}


def _traub_inf_v(name, v):
    a, b = _TRAUB_RATE_V[name]
    return a(v) / (a(v) + b(v))


def _traub_inf_q(xi):
    return alpha_q_traub(xi) / (alpha_q_traub(xi) + beta_q_traub(xi))


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
    M, N, C, A, H, B, S, R, Q, XI = [states[i] for i in range(10)]
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
    M, N, C, A, H, B, S, R, Q, XI = [states[i] for i in range(10)]
    dv = traub_dv(p, u_t, v, states)

    dM = _traub_dstate_v("M", v, M)
    dN = _traub_dstate_v("N", v, N)
    dC = _traub_dstate_v("C", v, C)
    dA = _traub_dstate_v("A", v, A)
    dH = _traub_dstate_v("H", v, H)
    dB = _traub_dstate_v("B", v, B)
    dS = _traub_dstate_v("S", v, S)
    dR = _traub_dstate_v("R", v, R)
    dQ = alpha_q_traub(XI) * (1.0 - Q) - beta_q_traub(XI) * Q
    i_ca = p.g_Ca * S * S * R * (v - p.V_Ca)
    dXI = -p.phi_area * i_ca - p.Beta * XI

    return dv, jnp.stack([dM, dN, dC, dA, dH, dB, dS, dR, dQ, dXI])


# 状態順序: [M, N, C, A, H, B, S, R, Q, XI]。hybrid の学習ゲートは「先頭 n 本」規約
# (TrainSource.gate) なので、学習するものを先頭へ並べる = この順序が分割位置を決める。
TRAUB_STATE_NAMES = ["M", "N", "C", "A", "H", "B", "S", "R", "Q", "XI"]

# hybrid での学習/physics 分割は 2 通りあり、preset (meta.physics_type) が選ぶ。
# どちらも「学習ゲート = 先頭 n 本、残りが extra」という TrainSource の規約に乗る
# ので、状態順序が分割位置をそのまま決める。
#
# 既定: 純電位依存 8 ゲートを学習し、Ca サブ系の濃度 XI と Q だけ physics。
TRAUB_LEARNED_GATE_NAMES = ["M", "N", "C", "A", "H", "B", "S", "R"]
TRAUB_EXTRA_GATE_NAMES = ["Q", "XI"]
# S,R も physics へ回す版: i_ca = g_Ca·S²·R が XI の積分器を駆動するため、S,R の
# decode 誤差が S の 2 乗で増幅され XI へバイアスとして蓄積する。XI は i_kahp/i_kc
# (AHP = バースト終息機構) を駆動するので、そのずれが発火パターンの崩壊に直結する。
# S,R は純電位依存 (レート関数だけで解ける = params-free) なので physics 側へ回して
# も 1 サロゲートを任意 comp へ移植できる性質は保たれる。
TRAUB_SR_LEARNED_GATE_NAMES = ["M", "N", "C", "A", "H", "B"]
TRAUB_SR_EXTRA_GATE_NAMES = ["S", "R", "XI", "Q"]


def traub_calcium_step(p: TraubParams, v, gates8, extra):
    """Ca サブ系 physics step。gates8=decode 済 [M,N,C,A,H,B,S,R]、extra=[Q,XI]。
    traub_dv 用の全 10 状態と d(extra)=[dQ,dXI] を返す。"""
    Q, XI = extra[0], extra[1]
    S, R = gates8[6], gates8[7]
    i_ca = p.g_Ca * S * S * R * (v - p.V_Ca)
    dQ = alpha_q_traub(XI) * (1.0 - Q) - beta_q_traub(XI) * Q
    dXI = -p.phi_area * i_ca - p.Beta * XI
    return jnp.concatenate([gates8, extra]), jnp.stack([dQ, dXI])


def traub_extra_inits(p: TraubParams) -> list[float]:
    """[Q, XI] 初期値 (V_LEAK 定常)。params のみ依存 → load 後も再現可。"""
    xi = _traub_xi_init(p)
    return [float(_traub_inf_q(jnp.asarray(xi))), xi]


def traub_sr_calcium_step(p: TraubParams, v, gates6, extra):
    """S,R も physics で解く版。gates6=[M,N,C,A,H,B]、extra=[S,R,XI,Q]。"""
    S, R, XI, Q = extra[0], extra[1], extra[2], extra[3]
    i_ca = p.g_Ca * S * S * R * (v - p.V_Ca)
    dXI = -p.phi_area * i_ca - p.Beta * XI
    dQ = alpha_q_traub(XI) * (1.0 - Q) - beta_q_traub(XI) * Q
    full = jnp.concatenate([gates6, jnp.stack([S, R, Q, XI])])
    return full, jnp.stack(
        [_traub_dstate_v("S", v, S), _traub_dstate_v("R", v, R), dXI, dQ]
    )


def traub_sr_extra_inits(p: TraubParams) -> list[float]:
    """[S, R, XI, Q] 初期値 (V_LEAK 定常)。"""
    return [
        float(_traub_inf_v("S", p.V_LEAK)),
        float(_traub_inf_v("R", p.V_LEAK)),
        _traub_xi_init(p),
        float(_traub_inf_q(jnp.asarray(_traub_xi_init(p)))),
    ]


def _traub_xi_init(p: TraubParams) -> float:
    """V_LEAK 定常 (dXI/dt = 0) の Ca 濃度。"""
    s = _traub_inf_v("S", p.V_LEAK)
    r = _traub_inf_v("R", p.V_LEAK)
    return float(-p.phi_area * (p.g_Ca * s * s * r * (p.V_LEAK - p.V_Ca)) / p.Beta)


def _traub_state_inits(p: TraubParams) -> list[float]:
    v0 = p.V_LEAK
    m = float(_traub_inf_v("M", v0))
    n = float(_traub_inf_v("N", v0))
    c = float(_traub_inf_v("C", v0))
    a = float(_traub_inf_v("A", v0))
    h = float(_traub_inf_v("H", v0))
    b = float(_traub_inf_v("B", v0))
    s = float(_traub_inf_v("S", v0))
    r = float(_traub_inf_v("R", v0))
    # i_Ca(v0) を用いて XI 初期値を求める (定常: dXI/dt = 0)
    i_ca0 = p.g_Ca * s * s * r * (v0 - p.V_Ca)
    xi = float(-p.phi_area * i_ca0 / p.Beta)
    q = float(_traub_inf_q(jnp.asarray(xi)))
    return [m, n, c, a, h, b, s, r, q, xi]


def _traub_dstate_v(name, v, x):
    a, b = _TRAUB_RATE_V[name]
    return a(v) * (1.0 - x) - b(v) * x


# レート関数の演算コスト。_traub_u の pm=1、lin_exp_form の (exp=1, pm=1, div=1) を
# 含む。jnp.where の分岐は両枝とも評価されるため両方を積算 (比較自体は無視)。
TRAUB_RATE_COST_MAP: dict[str, OpCost] = {
    "alpha_m_traub": OpCost(exp=1, div=2, pm=3, mul=1),
    "beta_m_traub": OpCost(exp=1, div=2, pm=3, mul=1),
    "alpha_s_traub": OpCost(exp=1, div=1, pm=3, mul=1),
    "beta_s_traub": OpCost(exp=1, div=2, pm=3, mul=1),
    "alpha_n_traub": OpCost(exp=1, div=2, pm=3, mul=1),
    "beta_n_traub": OpCost(exp=1, div=1, pm=2, mul=1),
    "alpha_c_traub": OpCost(exp=2, div=4, pm=6, mul=1),
    "beta_c_traub": OpCost(exp=3, div=5, pm=10, mul=2),  # alpha_c 再計算を含む
    "alpha_a_traub": OpCost(exp=1, div=2, pm=3, mul=1),
    "beta_a_traub": OpCost(exp=1, div=2, pm=3, mul=1),
    "alpha_h_traub": OpCost(exp=1, div=1, pm=2, mul=1),
    "beta_h_traub": OpCost(exp=1, div=2, pm=3),
    "alpha_r_traub": OpCost(exp=1, div=2, pm=2),
    "beta_r_traub": OpCost(exp=1, div=2, pm=3),  # alpha_r 再計算を含む
    "alpha_b_traub": OpCost(exp=1, div=1, pm=2, mul=1),
    "beta_b_traub": OpCost(exp=1, div=2, pm=3),
}


# Ca サブ系 physics の演算コスト (i_ca + dXI + dQ)。
TRAUB_CA_COST = (
    OpCost(pm=1, mul=4)  # i_ca = g_Ca·S·S·R·(v-V_Ca)
    + OpCost(pm=1, mul=2)  # dXI = -phi_area·i_ca - Beta·XI
    + OpCost(pm=2, mul=3)  # dQ = alpha_q·(1-Q) - beta_q·Q (min/const は無視)
)

# S,R も physics で解く版のコスト = 上記 + S,R のレート積分 2 本。
TRAUB_SR_CA_COST = (
    TRAUB_CA_COST
    + TRAUB_RATE_COST_MAP["alpha_s_traub"]
    + TRAUB_RATE_COST_MAP["beta_s_traub"]
    + OpCost(pm=2, mul=2)  # dS = alpha_s·(1-S) - beta_s·S
    + TRAUB_RATE_COST_MAP["alpha_r_traub"]
    + TRAUB_RATE_COST_MAP["beta_r_traub"]
    + OpCost(pm=2, mul=2)  # dR = alpha_r·(1-R) - beta_r·R
)


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


def traub_inits(p: TraubParams) -> list[float]:
    """[V, *TRAUB_STATE_NAMES] 初期状態。静止電位 = 自身の V_LEAK。"""
    return [p.V_LEAK] + _traub_state_inits(p)


TRAUB_TYPE = CompartmentType(
    name="traub",
    kernel=calc_traub_channel,
    param_cls=TraubParams,
    gate_names=TRAUB_STATE_NAMES,
    inits=traub_inits,
    opcost=OpCost(),  # TODO: 実測 or 積算
)
