#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
import cython

from neurosurrogate.dataset_utils._base import run_simulation, CHUNK_SIZE

cdef extern from "hh.c" nogil:
    cdef enum:
        N_VARS_HH = 4
        N_VARS_THREE_COMP = 6

    ctypedef struct HH_params:
        double E_REST
        double C
        double G_LEAK
        double E_LEAK
        double G_NA
        double E_NA
        double G_K
        double E_K
        double DT

    ctypedef struct ThreeComp_params:
        HH_params hh
        double G_12
        double G_23

    void initialize_hh(double *var, HH_params* p)
    void solve_euler_hh(double *var, double I_inj, HH_params* p)
    void threecomp_initialize_unified(double *var, ThreeComp_params* p)
    void solve_euler_threecomp_unified(double *var, double i_inj, ThreeComp_params* p)

# --- Python-facing parameter classes ---

cdef class HH_Params:
    cdef HH_params _params
    def __cinit__(self, E_REST=-65.0, C=1.0, G_LEAK=0.3, E_LEAK=10.6-65.0, 
                  G_NA=120.0, E_NA=115.0-65.0, G_K=36.0, E_K=-12.0-65.0, DT=0.01):
        self._params.E_REST = E_REST
        self._params.C = C
        self._params.G_LEAK = G_LEAK
        self._params.E_LEAK = E_LEAK
        self._params.G_NA = G_NA
        self._params.E_NA = E_NA
        self._params.G_K = G_K
        self._params.E_K = E_K
        self._params.DT = DT
    
    @property
    def DT(self):
        return self._params.DT

cdef class ThreeComp_Params:
    cdef ThreeComp_params _params
    def __cinit__(self, E_REST=-65.0, C=1.0, G_LEAK=0.3, E_LEAK=10.6-65.0, 
                  G_NA=120.0, E_NA=115.0-65.0, G_K=36.0, E_K=-12.0-65.0, DT=0.01,
                  G_12=0.1, G_23=0.05):
        self._params.hh.E_REST = E_REST
        self._params.hh.C = C
        self._params.hh.G_LEAK = G_LEAK
        self._params.hh.E_LEAK = E_LEAK
        self._params.hh.G_NA = G_NA
        self._params.hh.E_NA = E_NA
        self._params.hh.G_K = G_K
        self._params.hh.E_K = E_K
        self._params.hh.DT = DT
        self._params.G_12 = G_12
        self._params.G_23 = G_23

    @property
    def DT(self):
        return self._params.hh.DT
    
    @property
    def G_12(self):
        return self._params.G_12
    
    @property
    def G_23(self):
        return self._params.G_23

# --- Main simulation functions ---

def hh_simulate(fp, params):
    cdef HH_Params py_params
    py_params = params
    cdef HH_params* c_params = &py_params._params

    def get_I_ext_chunk(start, end):
        return fp["I_ext"][start:end]
        
    cdef double var[N_VARS_HH]
    cdef int NT = np.shape(fp["I_ext"])[0]
    
    def _initialize():
        initialize_hh(var, c_params)
        
    dset_shape = (NT, N_VARS_HH)
    dset_chunks = (CHUNK_SIZE, N_VARS_HH)
    
    def step_chunk(chunk_view, I_ext_chunk_view):
        cdef int i, n
        cdef int chunk_size = chunk_view.shape[0]
        cdef int n_vars = chunk_view.shape[1]
        for i in range(chunk_size):
            I_ext_val = I_ext_chunk_view[i]
            solve_euler_hh(&var[0], I_ext_val, c_params)
            for n in range(n_vars):
                chunk_view[i, n] = var[n]
        
    run_simulation(fp, NT, dset_shape, dset_chunks, _initialize, step_chunk, get_I_ext_chunk)

def threecomp_simulate(fp, params):
    cdef ThreeComp_Params py_params
    py_params = params
    cdef ThreeComp_params* c_params = &py_params._params

    def get_I_ext_chunk(start, end):
        return fp["I_ext"][start:end]
        
    cdef double var[N_VARS_THREE_COMP]
    cdef int NT = np.shape(fp["I_ext"])[0]
    
    def _initialize():
        threecomp_initialize_unified(var, c_params)
        
    dset_shape = (NT, N_VARS_THREE_COMP)
    dset_chunks = (CHUNK_SIZE, N_VARS_THREE_COMP)
    
    def step_chunk(chunk_view, I_ext_chunk_view):
        cdef int i, n
        cdef int chunk_size = chunk_view.shape[0]
        cdef int n_vars = chunk_view.shape[1]
        for i in range(chunk_size):
            I_ext_val = I_ext_chunk_view[i]
            solve_euler_threecomp_unified(&var[0], I_ext_val, c_params)
            for n in range(n_vars):
                chunk_view[i, n] = var[n]
        
    run_simulation(fp, NT, dset_shape, dset_chunks, _initialize, step_chunk, get_I_ext_chunk)