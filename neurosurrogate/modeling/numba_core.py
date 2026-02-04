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
def initialize_hh(p):
    v = p.E_REST
    v_rel = v - p.E_REST
    return np.array([v, m0(v_rel), h0(v_rel), n0(v_rel)])


@njit
def initialize_hh3(p):
    init_hh = initialize_hh(p.hh)
    init_passiv_comp = np.array([p.hh.E_REST, p.hh.E_REST])
    return np.concatenate((init_hh, init_passiv_comp))


@njit
def calc_deriv_hh(curr_x, u_t, model_args, dvar):
    p = model_args[0]

    v = curr_x[0]
    m = curr_x[1]
    h = curr_x[2]
    n = curr_x[3]

    v_rel = v - p.E_REST

    i_leak = p.G_LEAK * (v - p.E_LEAK)
    i_na = p.G_NA * m * m * m * h * (v - p.E_NA)
    i_k = p.G_K * n * n * n * n * (v - p.E_K)

    dvar[0] = (-i_leak - i_na - i_k + u_t) / p.C
    dvar[1] = (1.0 / tau_m(v_rel)) * (-m + m0(v_rel))
    dvar[2] = (1.0 / tau_h(v_rel)) * (-h + h0(v_rel))
    dvar[3] = (1.0 / tau_n(v_rel)) * (-n + n0(v_rel))


@njit
def calc_deriv_hh3(curr_x, u_t, model_args, dvar):
    p = model_args[0]

    v_soma = curr_x[0]
    v_pre = curr_x[4]
    v_post = curr_x[5]

    i_pre = p.G_12 * (v_pre - v_soma)
    i_post = p.G_23 * (v_soma - v_post)

    soma_model_args = (p.hh,)
    calc_deriv_hh(curr_x, i_pre - i_post, soma_model_args, dvar[:4])

    dvar[4] = (-p.hh.G_LEAK * (v_pre - p.hh.E_LEAK) - i_pre + u_t) / p.hh.C
    dvar[5] = (-p.hh.G_LEAK * (v_post - p.hh.E_LEAK) + i_post) / p.hh.C


@njit
def calc_deriv_sindy(curr_x, u_t, model_args, dvar):
    xi_matrix, compute_theta = model_args
    theta = compute_theta(curr_x[0], curr_x[1], u_t)
    dvar[:] = xi_matrix @ theta


@njit
def calc_deriv_sindy_hh3(curr_x, u_t, model_args, dvar):
    xi_matrix, params, compute_theta = model_args

    v_soma = curr_x[0]
    latent = curr_x[1]
    v_pre = curr_x[2]
    v_post = curr_x[3]

    # 1. コンパートメント間の電流計算
    I_pre = params.G_12 * (v_pre - v_soma)
    I_post = params.G_23 * (v_soma - v_post)

    # 2. SINDyモデルによる微分値の計算 (dx = Xi @ Theta)
    theta = compute_theta(v_soma, latent, I_pre - I_post)

    dvar[0] = xi_matrix[0] @ theta
    dvar[1] = xi_matrix[1] @ theta
    dvar[2] = (
        -params.hh.G_LEAK * (v_pre - params.hh.E_LEAK) - I_pre + u_t
    ) / params.hh.C
    dvar[3] = (-params.hh.G_LEAK * (v_post - params.hh.E_LEAK) + I_post) / params.hh.C


@njit
def generic_euler_solver(deriv_func, init, u, dt, model_args):
    n_steps = len(u)
    n_vars = len(init)
    x_history = np.zeros((n_steps, n_vars))

    curr_x = init.copy()
    x_history[0] = curr_x
    dvar = np.zeros(n_vars)

    for t in range(n_steps - 1):
        # 微分計算関数の呼び出し。model_argsはタプル。
        deriv_func(curr_x, u[t], model_args, dvar)

        # 状態更新
        for i in range(n_vars):
            curr_x[i] += dvar[i] * dt
        x_history[t + 1] = curr_x

    return x_history


PARAMS_REGISTRY = {
    "hh": HH_Params_numba,
    "hh3": ThreeComp_Params_numba,
}

SIMULATER_CONFIGS = {
    "hh": {
        "deriv_func": calc_deriv_hh,
        "features": ["V_soma", "M", "H", "N"],
        "init_func": lambda p: initialize_hh(p),
        "args_factory": lambda p, **kwargs: (p,),
    },
    "hh3": {
        "deriv_func": calc_deriv_hh3,
        "features": ["V_soma", "M", "H", "N", "V_pre", "V_post"],
        "init_func": lambda p: initialize_hh3(p),
        "args_factory": lambda p, **kwargs: (p,),
    },
}

SURROGATER_CONFIGS = {
    "hh": {
        "deriv_func": calc_deriv_sindy,
        "features": ["V_soma", "latent1"],
        "args_factory": lambda xi, p, func, **kwargs: (xi, func),
    },
    "hh3": {
        "deriv_func": calc_deriv_sindy_hh3,
        "features": ["V_soma", "latent1", "V_pre", "V_post"],
        "args_factory": lambda xi, p, func, **kwargs: (xi, p, func),
    },
}
