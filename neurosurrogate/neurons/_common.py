import jax.numpy as jnp


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
