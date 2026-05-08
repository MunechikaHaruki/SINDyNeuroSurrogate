import numpy as np
from numba import float64, njit
from numba.experimental import jitclass

from .calc_utils import Compartment
from .profiler import OpCost

Compartment


@njit
def lin_exp_form(x):
    condition = np.abs(x) < 1e-8

    # 特異点付近（テイラー展開）
    approx = 1.0 / (1.0 + x / 2.0 + x**2 / 6.0 + x**3 / 24.0)

    # 生の式（0除算を避けるための微小値 epsilon）
    # 分母が0だと計算が止まるので、一時的に1.0にしておき、後で where で捨てます
    denom = np.exp(x) - 1.0
    safe_denom = np.where(denom == 0, 1.0, denom)
    raw = x / safe_denom

    return np.where(condition, approx, raw)


@njit
def alpha_m(v):
    return lin_exp_form(2.5 - 0.1 * v)


@njit
def beta_m(v):
    return 4.0 * np.exp(-v / 18.0)


@njit
def alpha_h(v):
    return 0.07 * np.exp(-v / 20.0)


@njit
def beta_h(v):
    return 1.0 / (np.exp(3.0 - 0.1 * v) + 1.0)


@njit
def alpha_n(v):
    return 0.1 * lin_exp_form(1 - 0.1 * v)


@njit
def beta_n(v):
    return 0.125 * np.exp(-v / 80.0)


FUNC_COST_MAP: dict[str, OpCost] = {
    "alpha_m": OpCost(exp=1, div=1, pm=2, mul=2),
    "beta_m": OpCost(exp=1, div=1, pm=1, mul=1),
    "alpha_h": OpCost(exp=1, div=1, pm=1, mul=1),
    "beta_h": OpCost(exp=1, div=1, pm=2, mul=1),
    "alpha_n": OpCost(exp=1, div=1, pm=2, mul=2),
    "beta_n": OpCost(exp=1, div=1, pm=1, mul=1),
}


def _get_original_hh_cost():
    """
    提供された calc_deriv_hh / hh3 のコードを静的にトレースした演算コスト。
    """
    gate_cost = OpCost()
    # 1. alpha/beta (6個分)
    for op in FUNC_COST_MAP.values():
        gate_cost = gate_cost + op

    gating_cost = OpCost(pm=6, div=6)
    # 3. calc_hh_channel 内部
    channel_cost = OpCost(
        pm=1 + 1 + 1 + 1 + 4 + 6,  # v_rel, i_leak, i_na, i_k, dvar[0], dvar[1-3]×2
        mul=1 + 5 + 5 + 3,  # i_leak, i_na, i_k, dvar[1-3]
        div=1 + 3,  # dvar[0], dvar[1-3]
    )
    return gate_cost + gating_cost + channel_cost


HH_COST = _get_original_hh_cost()


@jitclass(
    [
        ("E_REST", float64),
        ("C", float64),
        ("G_LEAK", float64),
        ("E_LEAK", float64),
        ("G_NA", float64),
        ("E_NA", float64),
        ("G_K", float64),
        ("E_K", float64),
    ]
)
class HH_Params_numba:
    def __init__(self):
        self.E_REST = -65.0
        self.C = 1.0
        self.G_LEAK = 0.3
        self.E_LEAK = 10.6 - 65.0
        self.G_NA = 120.0
        self.E_NA = 115.0 - 65.0
        self.G_K = 36.0
        self.E_K = -12.0 - 65.0


@njit
def calc_hh_channel(p, u_t, v, curr_gate, dvar_gate):
    m = curr_gate[0]
    h = curr_gate[1]
    n = curr_gate[2]
    v_rel = v - p.E_REST

    i_leak = p.G_LEAK * (v - p.E_LEAK)
    i_na = p.G_NA * m * m * m * h * (v - p.E_NA)
    i_k = p.G_K * n * n * n * n * (v - p.E_K)

    dv = (-i_leak - i_na - i_k + u_t) / p.C
    dvar_gate[0] = alpha_m(v_rel) * (1.0 - m) - beta_m(v_rel) * m
    dvar_gate[1] = alpha_h(v_rel) * (1.0 - h) - beta_h(v_rel) * h
    dvar_gate[2] = alpha_n(v_rel) * (1.0 - n) - beta_n(v_rel) * n
    return dv


@njit
def calc_passive_channel(p, u_t, v):
    return (-p.G_LEAK * (v - p.E_LEAK) + u_t) / p.C


V_REL = (-65) - (-65)  # V_INIT - E_REST

COMPARTMENT_TEMPLATES = {
    "hh": Compartment(
        gate_inits=[
            alpha_m(V_REL) / (alpha_m(V_REL) + beta_m(V_REL)),
            alpha_h(V_REL) / (alpha_h(V_REL) + beta_h(V_REL)),
            alpha_n(V_REL) / (alpha_n(V_REL) + beta_n(V_REL)),
        ],
        gate_names=["M", "H", "N"],
    ),
    "passive": Compartment(gate_inits=[], gate_names=[]),
}
