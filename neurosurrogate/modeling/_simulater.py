import numpy as np
from numba import float64, njit
from numba.experimental import jitclass

# jitclass for HH parameters
hh_params_spec = [
    ("E_REST", float64),
    ("C", float64),
    ("G_LEAK", float64),
    ("E_LEAK", float64),
    ("G_NA", float64),
    ("E_NA", float64),
    ("G_K", float64),
    ("E_K", float64),
    ("DT", float64),
]


@jitclass(hh_params_spec)
class HH_Params_numba:
    def __init__(self, params_dict):
        self.E_REST = params_dict.get("E_REST", -65.0)
        self.C = params_dict.get("C", 1.0)
        self.G_LEAK = params_dict.get("G_LEAK", 0.3)
        self.E_LEAK = params_dict.get("E_LEAK", 10.6 - 65.0)
        self.G_NA = params_dict.get("G_NA", 120.0)
        self.E_NA = params_dict.get("E_NA", 115.0 - 65.0)
        self.G_K = params_dict.get("G_K", 36.0)
        self.E_K = params_dict.get("E_K", -12.0 - 65.0)


# jitclass for Three-compartment model parameters
threecomp_params_spec = [
    ("hh", HH_Params_numba.class_type.instance_type),
    ("G_12", float64),
    ("G_23", float64),
]


@jitclass(threecomp_params_spec)
class ThreeComp_Params_numba:
    def __init__(self, params_dict):
        self.hh = HH_Params_numba(params_dict)
        self.G_12 = params_dict.get("G_12", 1)
        self.G_23 = params_dict.get("G_23", 0.7)


# G_12=0.1, G_23=0.05


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


@njit
def dmdt(v_rel, m):
    return (1.0 / tau_m(v_rel)) * (-m + m0(v_rel))


@njit
def dhdt(v_rel, h):
    return (1.0 / tau_h(v_rel)) * (-h + h0(v_rel))


@njit
def dndt(v_rel, n):
    return (1.0 / tau_n(v_rel)) * (-n + n0(v_rel))


@njit
def dvdt(i_leak, i_na, i_k, i_ext, c):
    return (-i_leak - i_na - i_k + i_ext) / c


@njit
def initialize_hh(var, p):
    v = p.E_REST
    var[0] = v
    v_rel = v - p.E_REST
    var[1] = m0(v_rel)
    var[2] = h0(v_rel)
    var[3] = n0(v_rel)


@njit
def solve_euler_hh(var, i_inj, p, DT):
    v = var[0]
    m = var[1]
    h = var[2]
    n = var[3]

    v_rel = v - p.E_REST

    i_leak = p.G_LEAK * (v - p.E_LEAK)
    i_na = p.G_NA * m * m * m * h * (v - p.E_NA)
    i_k = p.G_K * n * n * n * n * (v - p.E_K)
    var[0] += dvdt(i_leak, i_na, i_k, i_inj, p.C) * DT
    var[1] += dmdt(v_rel, m) * DT
    var[2] += dhdt(v_rel, h) * DT
    var[3] += dndt(v_rel, n) * DT


@njit
def hh_simulate_numba(i_ext, p, DT):
    n_vars = 4
    nt = len(i_ext)
    results = np.zeros((nt, n_vars))
    var = np.zeros(n_vars)
    initialize_hh(var, p)
    for i in range(nt):
        solve_euler_hh(var, i_ext[i], p, DT)
        results[i, :] = var
    return results


@njit
def threecomp_initialize_unified(var, p):
    initialize_hh(var, p.hh)
    var[4] = p.hh.E_REST
    var[5] = p.hh.E_REST


@njit
def solve_euler_threecomp_unified(var, i_inj, p, DT):
    v_soma = var[0]
    v_pre = var[4]
    v_post = var[5]

    i_pre = p.G_12 * (v_pre - v_soma)
    i_post = p.G_23 * (v_soma - v_post)

    solve_euler_hh(var, i_pre - i_post, p.hh, DT)

    var[4] += (-p.hh.G_LEAK * (v_pre - p.hh.E_LEAK) - i_pre + i_inj) / p.hh.C * DT
    var[5] += (-p.hh.G_LEAK * (v_post - p.hh.E_LEAK) + i_post) / p.hh.C * DT


@njit
def hh3_simulate_numba(i_ext, p, DT):
    n_vars = 6
    nt = len(i_ext)
    results = np.zeros((nt, n_vars))
    var = np.zeros(n_vars)
    threecomp_initialize_unified(var, p)
    for i in range(nt):
        solve_euler_threecomp_unified(var, i_ext[i], p, DT)
        results[i, :] = var
    return results
