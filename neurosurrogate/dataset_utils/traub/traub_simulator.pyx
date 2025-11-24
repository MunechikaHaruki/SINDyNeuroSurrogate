#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
import cython

from neurosurrogate.dataset_utils._base import run_simulation, CHUNK_SIZE

# NumPy C-APIの初期化
np.import_array()

cdef extern from "traub.h" nogil:
    cdef enum:
        N_VARS_TRAUB
        NC_TRAUB
    
    void initialize_traub(double var[N_VARS_TRAUB][NC_TRAUB], double I_inj[NC_TRAUB], double g_comp[NC_TRAUB][2])
    void solve_euler_traub(double var[N_VARS_TRAUB][NC_TRAUB], double I_inj[NC_TRAUB], double g_comp[NC_TRAUB][2])

cdef void _step_chunk_nogil(double[:, :, :] chunk_view, double[:, :] I_ext_chunk_view, double var[N_VARS_TRAUB][NC_TRAUB], double g_comp[NC_TRAUB][2]) nogil:
    cdef int i, n, c
    cdef int chunk_size = chunk_view.shape[0]
    for i in range(chunk_size):
        solve_euler_traub(var, &I_ext_chunk_view[i][0], g_comp)
        for n in range(N_VARS_TRAUB):
            for c in range(NC_TRAUB):
                chunk_view[i, n, c] = var[n][c]

# --- メインのシミュレーション関数 ---
def traub_simulate(fp,iter):
    cdef double var[N_VARS_TRAUB][NC_TRAUB]
    cdef double I_inj[NC_TRAUB]
    cdef double g_comp[NC_TRAUB][2]
    cdef int NT = iter
    
    def _initialize():
        initialize_traub(var, I_inj, g_comp)

    dset_shape = (NT, N_VARS_TRAUB, NC_TRAUB)
    dset_chunks = (CHUNK_SIZE, N_VARS_TRAUB, NC_TRAUB)

    def get_I_ext_chunk(start, end):
        return fp["I_ext"][start:end, :]

    def step_chunk(chunk_view, I_ext_chunk_view):
        _step_chunk_nogil(chunk_view, I_ext_chunk_view, var, g_comp)

    run_simulation(fp, NT, dset_shape, dset_chunks, _initialize, step_chunk, get_I_ext_chunk)
