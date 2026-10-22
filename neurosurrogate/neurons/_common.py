import json
from pathlib import Path

import jax.numpy as jnp

from ..core.opcost import OpCost


def _lin_exp_form(x):
    """x / (e^x − 1)。|x| < 1e-4 はテイラー展開 (TS の bernoulli と同じ)。"""
    small = jnp.abs(x) < 1e-4
    safe = jnp.where(small, 1.0, x)
    return jnp.where(
        small,
        1.0 - x / 2.0 + x**2 / 12.0 - x**4 / 720.0,
        safe / jnp.expm1(safe),
    )


def inf_ode(alpha, beta):
    def inf(v):
        return alpha(v) / (alpha(v) + beta(v))

    return inf


def gate_ode(alpha, beta):
    def dxdt(v, x):
        return alpha(v) * (1.0 - x) - beta(v) * x

    return dxdt


def read_spec(name):
    """隣の <name>.json を読む。rates はレートの表 {ゲート: {alpha, beta}}。"""
    return json.loads(Path(__file__).with_name(f"{name}.json").read_text())


def _exp_form(a, z):
    return a * jnp.exp(z)


def _sigmoid_form(a, z):
    return a / (1.0 + jnp.exp(z))


def _linexp_form(a, z):
    return a * _lin_exp_form(z)


# JSON の form → (A, z) の式と、その演算コスト (_lin_exp_form は exp=1, pm=1, div=1)。
_FORMS = {"exp": _exp_form, "sigmoid": _sigmoid_form, "linexp": _linexp_form}
_FORM_COSTS = {
    "exp": OpCost(exp=1, mul=1),
    "sigmoid": OpCost(exp=1, pm=1, div=1),
    "linexp": OpCost(exp=1, pm=1, div=1, mul=1),
}


def form_rate(spec):
    """{form, A, theta, s} → u ↦ A·form((θ − u)/s)。u は原著の基準電位からの電位。"""
    form = _FORMS[spec["form"]]

    def rate(u):
        return form(spec["A"], (spec["theta"] - u) / spec["s"])

    return rate


def form_rate_cost(spec):
    """form_rate の演算コスト。(θ − u)/s の pm=1, div=1 と形の分。"""
    return OpCost(pm=1, div=1) + _FORM_COSTS[spec["form"]]
