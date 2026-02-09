import numpy as np
from numba import njit


@njit
def alpha_m(v):
    return (2.5 - 0.1 * v) / (np.exp(2.5 - 0.1 * v) - 1.0)


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
    return (0.1 - 0.01 * v) / (np.exp(1 - 0.1 * v) - 1.0)


@njit
def beta_n(v):
    return 0.125 * np.exp(-v / 80.0)


@njit
def m0(v_rel):
    a_m = alpha_m(v_rel)
    b_m = beta_m(v_rel)
    return a_m / (a_m + b_m)


@njit
def h0(v_rel):
    a_h = alpha_h(v_rel)
    b_h = beta_h(v_rel)
    return a_h / (a_h + b_h)


@njit
def n0(v_rel):
    a_n = alpha_n(v_rel)
    b_n = beta_n(v_rel)
    return a_n / (a_n + b_n)


@njit
def tau_m(v_rel):
    return 1.0 / (alpha_m(v_rel) + beta_m(v_rel))


@njit
def tau_h(v_rel):
    return 1.0 / (alpha_h(v_rel) + beta_h(v_rel))


@njit
def tau_n(v_rel):
    return 1.0 / (alpha_n(v_rel) + beta_n(v_rel))
