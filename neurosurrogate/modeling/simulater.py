import numpy as np
from numba import float64, jit
from numba.experimental import jitclass
from tqdm import tqdm

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
    def __init__(
        self,
        E_REST=-65.0,
        C=1.0,
        G_LEAK=0.3,
        E_LEAK=10.6 - 65.0,
        G_NA=120.0,
        E_NA=115.0 - 65.0,
        G_K=36.0,
        E_K=-12.0 - 65.0,
        DT=0.01,
    ):
        self.E_REST = E_REST
        self.C = C
        self.G_LEAK = G_LEAK
        self.E_LEAK = E_LEAK
        self.G_NA = G_NA
        self.E_NA = E_NA
        self.G_K = G_K
        self.E_K = E_K
        self.DT = DT


# jitclass for Three-compartment model parameters
threecomp_params_spec = [
    ("hh", HH_Params_numba.class_type.instance_type),
    ("G_12", float64),
    ("G_23", float64),
]


@jitclass(threecomp_params_spec)
class ThreeComp_Params_numba:
    def __init__(self, hh, G_12=0.1, G_23=0.05):
        self.hh = hh
        self.G_12 = G_12
        self.G_23 = G_23

    @property
    def DT(self):
        return self.hh.DT


@jit(nopython=True)
def alpha_m(v, p):
    return (2.5 - 0.1 * (v - p.E_REST)) / (np.exp(2.5 - 0.1 * (v - p.E_REST)) - 1.0)


@jit(nopython=True)
def beta_m(v, p):
    return 4.0 * np.exp(-(v - p.E_REST) / 18.0)


@jit(nopython=True)
def alpha_h(v, p):
    return 0.07 * np.exp(-(v - p.E_REST) / 20.0)


@jit(nopython=True)
def beta_h(v, p):
    return 1.0 / (np.exp(3.0 - 0.1 * (v - p.E_REST)) + 1.0)


@jit(nopython=True)
def alpha_n(v, p):
    return (0.1 - 0.01 * (v - p.E_REST)) / (np.exp(1 - 0.1 * (v - p.E_REST)) - 1.0)


@jit(nopython=True)
def beta_n(v, p):
    return 0.125 * np.exp(-(v - p.E_REST) / 80.0)


@jit(nopython=True)
def m0(v, p):
    return alpha_m(v, p) / (alpha_m(v, p) + beta_m(v, p))


@jit(nopython=True)
def h0(v, p):
    return alpha_h(v, p) / (alpha_h(v, p) + beta_h(v, p))


@jit(nopython=True)
def n0(v, p):
    return alpha_n(v, p) / (alpha_n(v, p) + beta_n(v, p))


@jit(nopython=True)
def tau_m(v, p):
    return 1.0 / (alpha_m(v, p) + beta_m(v, p))


@jit(nopython=True)
def tau_h(v, p):
    return 1.0 / (alpha_h(v, p) + beta_h(v, p))


@jit(nopython=True)
def tau_n(v, p):
    return 1.0 / (alpha_n(v, p) + beta_n(v, p))


@jit(nopython=True)
def dmdt(v, m, p):
    return (1.0 / tau_m(v, p)) * (-m + m0(v, p))


@jit(nopython=True)
def dhdt(v, h, p):
    return (1.0 / tau_h(v, p)) * (-h + h0(v, p))


@jit(nopython=True)
def dndt(v, n, p):
    return (1.0 / tau_n(v, p)) * (-n + n0(v, p))


@jit(nopython=True)
def dvdt(v, m, h, n, i_ext, p):
    return (
        -p.G_LEAK * (v - p.E_LEAK)
        - p.G_NA * m * m * m * h * (v - p.E_NA)
        - p.G_K * n * n * n * n * (v - p.E_K)
        + i_ext
    ) / p.C


@jit(nopython=True)
def initialize_hh(var, p):
    v = p.E_REST
    var[0] = v
    var[1] = m0(v, p)
    var[2] = h0(v, p)
    var[3] = n0(v, p)


@jit(nopython=True)
def solve_euler_hh(var, i_inj, p):
    v = var[0]
    m = var[1]
    h = var[2]
    n = var[3]
    var[0] += dvdt(v, m, h, n, i_inj, p) * p.DT
    var[1] += dmdt(v, m, p) * p.DT
    var[2] += dhdt(v, h, p) * p.DT
    var[3] += dndt(v, n, p) * p.DT


@jit(nopython=True)
def hh_simulate_numba(i_ext, p):
    n_vars = 4
    nt = len(i_ext)
    results = np.zeros((nt, n_vars))
    var = np.zeros(n_vars)
    initialize_hh(var, p)
    for i in range(nt):
        solve_euler_hh(var, i_ext[i], p)
        results[i, :] = var
    return results


@jit(nopython=True)
def threecomp_initialize_unified(var, p):
    initialize_hh(var, p.hh)
    var[4] = p.hh.E_REST
    var[5] = p.hh.E_REST


@jit(nopython=True)
def solve_euler_threecomp_unified(var, i_inj, p):
    v_soma = var[0]
    v_pre = var[4]
    v_post = var[5]

    i_pre = p.G_12 * (v_pre - v_soma)
    i_post = p.G_23 * (v_soma - v_post)

    solve_euler_hh(var, i_pre - i_post, p.hh)

    var[4] += (-p.hh.G_LEAK * (v_pre - p.hh.E_LEAK) - i_pre + i_inj) / p.hh.C * p.hh.DT
    var[5] += (-p.hh.G_LEAK * (v_post - p.hh.E_LEAK) + i_post) / p.hh.C * p.hh.DT


@jit(nopython=True)
def hh3_simulate_numba(i_ext, p):
    n_vars = 6
    nt = len(i_ext)
    results = np.zeros((nt, n_vars))
    var = np.zeros(n_vars)
    threecomp_initialize_unified(var, p)
    for i in range(nt):
        solve_euler_threecomp_unified(var, i_ext[i], p)
        results[i, :] = var
    return results


# Wrapper functions for numba simulators
CHUNK_SIZE = 10000


def hh_simulate_numba_wrapper(fp, params):
    i_ext = fp["I_ext"][:]
    results = hh_simulate_numba(i_ext, params)
    dset_shape = results.shape
    dset_chunks = (CHUNK_SIZE, dset_shape[1])
    dset = fp.create_dataset(
        "vars", shape=dset_shape, dtype="float64", chunks=dset_chunks
    )

    start_idx = 0
    with tqdm(total=len(i_ext)) as pbar:
        while start_idx < len(i_ext):
            end_idx = min(start_idx + CHUNK_SIZE, len(i_ext))
            dset[start_idx:end_idx] = results[start_idx:end_idx]
            pbar.update(end_idx - start_idx)
            start_idx = end_idx


def hh3_simulate_numba_wrapper(fp, params):
    i_ext = fp["I_ext"][:]
    results = hh3_simulate_numba(i_ext, params)
    dset_shape = results.shape
    dset_chunks = (CHUNK_SIZE, dset_shape[1])
    dset = fp.create_dataset(
        "vars", shape=dset_shape, dtype="float64", chunks=dset_chunks
    )

    start_idx = 0
    with tqdm(total=len(i_ext)) as pbar:
        while start_idx < len(i_ext):
            end_idx = min(start_idx + CHUNK_SIZE, len(i_ext))
            dset[start_idx:end_idx] = results[start_idx:end_idx]
            pbar.update(end_idx - start_idx)
            start_idx = end_idx
