import jax.numpy as jnp


def lin_exp_form(x):
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
