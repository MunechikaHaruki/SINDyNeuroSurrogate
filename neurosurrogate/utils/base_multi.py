# mypy: ignore-errors

from enum import IntEnum, auto

import numpy as np
import pysindy as ps


class Compartment(IntEnum):
    V = 0
    XI = auto()
    M = auto()
    S = auto()
    N = auto()
    C = auto()
    A = auto()
    H = auto()
    R = auto()
    B = auto()
    Q = auto()
    N_VARS = auto()


class NeuronModel:
    def __init__(self, nc: int = 19) -> None:
        self.NC = nc
        self.Cm = 3.0
        self.Ri = 0.1
        self.V_LEAK = -60.0
        self.V_Na = 115.0 + self.V_LEAK
        self.V_Ca = 140.0 + self.V_LEAK
        self.V_K = -15.0 + self.V_LEAK
        self.Beta = 0.075

        self.g_Na = np.array(
            [0, 0, 0, 0, 0, 20, 0, 15, 30, 15, 0, 20, 0, 0, 0, 0, 0, 0, 0]
        )
        self.g_K_DR = np.array(
            [0, 0, 0, 0, 0, 20, 0, 5, 15, 5, 0, 20, 0, 0, 0, 0, 0, 0, 0]
        )
        self.g_K_A = np.array([0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.g_K_C = np.array(
            [0, 5, 5, 10, 10, 10, 5, 20, 10, 20, 5, 15, 15, 15, 15, 15, 5, 5, 0]
        )
        self.g_K_AHP = np.array(
            [
                0,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0,
            ]
        )
        self.g_Ca = np.array(
            [0, 5, 5, 12, 12, 12, 5, 8, 4, 8, 5, 17, 17, 17, 10, 10, 5, 5, 0]
        )
        self.g_leak = np.full(self.NC, 0.1)

        self.phi = np.array(
            [
                7769,
                7769,
                7769,
                7769,
                7769,
                7769,
                7769,
                34530,
                17402,
                26404,
                5941,
                5941,
                5941,
                5941,
                5941,
                5941,
                5941,
                5941,
                5941,
            ]
        )
        self.rad = np.array(
            [
                2.89e-4,
                2.89e-4,
                2.89e-4,
                2.89e-4,
                2.89e-4,
                2.89e-4,
                2.89e-4,
                2.89e-4,
                4.23e-4,
                2.42e-4,
                2.42e-4,
                2.42e-4,
                2.42e-4,
                2.42e-4,
                2.42e-4,
                2.42e-4,
                2.42e-4,
                2.42e-4,
                2.42e-4,
            ]
        )
        self.len = np.array(
            [
                1.20e-2,
                1.20e-2,
                1.20e-2,
                1.20e-2,
                1.20e-2,
                1.20e-2,
                1.20e-2,
                1.20e-2,
                1.25e-2,
                1.10e-2,
                1.10e-2,
                1.10e-2,
                1.10e-2,
                1.10e-2,
                1.10e-2,
                1.10e-2,
                1.10e-2,
                1.10e-2,
                1.10e-2,
            ]
        )
        self.area = np.array(
            [
                2.188e-5,
                2.188e-5,
                2.188e-5,
                2.188e-5,
                2.188e-5,
                2.188e-5,
                2.188e-5,
                2.188e-5,
                3.320e-5,
                1.673e-5,
                1.673e-5,
                1.673e-5,
                1.673e-5,
                1.673e-5,
                1.673e-5,
                1.673e-5,
                1.673e-5,
                1.673e-5,
                1.673e-5,
            ]
        )

    def pir2(self, rad):
        return 3.141592 * (rad**2)

    def alpha_m(self, v):
        return (
            0.32
            * (13.1 - (v - self.V_LEAK))
            / (np.exp((13.1 - (v - self.V_LEAK)) / 4.0) - 1)
        )

    def beta_m(self, v):
        return (
            0.28
            * ((v - self.V_LEAK) - 40.1)
            / (np.exp(((v - self.V_LEAK) - 40.1) / 5.0) - 1)
        )

    def alpha_s(self, v):
        return 1.6 / (1 + np.exp(-0.072 * ((v - self.V_LEAK) - 65)))

    def beta_s(self, v):
        return (
            0.02
            * ((v - self.V_LEAK) - 51.1)
            / (np.exp(((v - self.V_LEAK) - 51.1) / 5.0) - 1)
        )

    def alpha_n(self, v):
        return (
            0.016
            * (35.1 - (v - self.V_LEAK))
            / (np.exp((35.1 - (v - self.V_LEAK)) / 5.0) - 1)
        )

    def beta_n(self, v):
        return 0.25 * np.exp((20 - (v - self.V_LEAK)) / 40.0)

    def alpha_c(self, v):
        v_shifted = v - self.V_LEAK
        return np.where(
            v <= (50 + self.V_LEAK),
            np.exp((v_shifted - 10) / 11.0 - (v_shifted - 6.5) / 27.0) / 18.975,
            2 * np.exp(-(v_shifted - 6.5) / 27.0),
        )

    def beta_c(self, v):
        return np.where(
            v <= (50 + self.V_LEAK),
            2 * np.exp(-((v - self.V_LEAK) - 6.5) / 27.0) - self.alpha_c(v),
            0,
        )

    def alpha_a(self, v):
        return (
            0.02
            * (13.1 - (v - self.V_LEAK))
            / (np.exp((13.1 - (v - self.V_LEAK)) / 10.0) - 1)
        )

    def beta_a(self, v):
        return (
            0.0175
            * ((v - self.V_LEAK) - 40.1)
            / (np.exp(((v - self.V_LEAK) - 40.1) / 10.0) - 1)
        )

    def alpha_h(self, v):
        return 0.128 * np.exp((17 - (v - self.V_LEAK)) / 18.0)

    def beta_h(self, v):
        return 4.0 / (1 + np.exp((40 - (v - self.V_LEAK)) / 5.0))

    def alpha_r(self, v):
        return np.where(
            v <= self.V_LEAK, 0.005, np.exp(-(v - self.V_LEAK) / 20.0) / 200.0
        )

    def beta_r(self, v):
        return np.where(v <= self.V_LEAK, 0, 0.005 - self.alpha_r(v))

    def alpha_b(self, v):
        return 0.0016 * np.exp((-13 - (v - self.V_LEAK)) / 18.0)

    def beta_b(self, v):
        return 0.05 / (1 + np.exp((10.1 - (v - self.V_LEAK)) / 5.0))

    def alpha_q(self, x):
        return np.minimum((0.2e-4) * x, 0.01)

    def beta_q(self, x):
        return 0.001

    def tau(self, var_enum, v_or_x):
        alpha_beta_map = {
            Compartment.M: (self.alpha_m, self.beta_m),
            Compartment.S: (self.alpha_s, self.beta_s),
            Compartment.N: (self.alpha_n, self.beta_n),
            Compartment.C: (self.alpha_c, self.beta_c),
            Compartment.A: (self.alpha_a, self.beta_a),
            Compartment.H: (self.alpha_h, self.beta_h),
            Compartment.R: (self.alpha_r, self.beta_r),
            Compartment.B: (self.alpha_b, self.beta_b),
            Compartment.Q: (self.alpha_q, self.beta_q),
        }
        alpha_func, beta_func = alpha_beta_map[var_enum]
        return 1.0 / (alpha_func(v_or_x) + beta_func(v_or_x))

    def inf(self, var_enum, v_or_x):
        alpha_beta_map = {
            Compartment.M: (self.alpha_m, self.beta_m),
            Compartment.S: (self.alpha_s, self.beta_s),
            Compartment.N: (self.alpha_n, self.beta_n),
            Compartment.C: (self.alpha_c, self.beta_c),
            Compartment.A: (self.alpha_a, self.beta_a),
            Compartment.H: (self.alpha_h, self.beta_h),
            Compartment.R: (self.alpha_r, self.beta_r),
            Compartment.B: (self.alpha_b, self.beta_b),
            Compartment.Q: (self.alpha_q, self.beta_q),
        }
        alpha_func, beta_func = alpha_beta_map[var_enum]
        return alpha_func(v_or_x) / (alpha_func(v_or_x) + beta_func(v_or_x))


class MultiCompartmentFuncFactory:
    def __init__(self, model: NeuronModel):
        self.model = model

    def create_base_functions(self):
        # Gate variable dynamics
        gate_funcs = []
        gate_names = []
        for j in range(Compartment.M, Compartment.N_VARS - 1):
            gate_funcs.append(
                lambda v, x, j=j: self.model.tau(j, v) * (-x + self.model.inf(j, v))
            )
            gate_names.append(
                lambda v,
                x,
                j=j: f"(inf_{Compartment(j).name}({v}) - {x}) / tau_{Compartment(j).name}({v})"
            )

        # Ion channel currents
        ion_current_funcs = [
            lambda v: self.model.g_leak * (v - self.model.V_LEAK),
            lambda v, m, h: self.model.g_Na * m**2 * h * (v - self.model.V_Na),
            lambda v, s, r: self.model.g_Ca * s**2 * r * (v - self.model.V_Ca),
            lambda v, n: self.model.g_K_DR * n * (v - self.model.V_K),
            lambda v, a, b: self.model.g_K_A * a * b * (v - self.model.V_K),
            lambda v, q: self.model.g_K_AHP * q * (v - self.model.V_K),
            lambda v, c, xi: self.model.g_K_C
            * c
            * np.minimum(1, xi / 250.0)
            * (v - self.model.V_K),
        ]
        ion_current_names = [
            lambda v: f"g_leak * ({v} - V_LEAK)",
            lambda v, m, h: f"g_Na * {m}^2 * {h} * ({v} - V_Na)",
            lambda v, s, r: f"g_Ca * {s}^2 * {r} * ({v} - V_Ca)",
            lambda v, n: f"g_K_DR * {n} * ({v} - V_K)",
            lambda v, a, b: f"g_K_A * {a} * {b} * ({v} - V_K)",
            lambda v, q: f"g_K_AHP * {q} * ({v} - V_K)",
            lambda v, c, xi: f"g_K_C * {c} * min(1, {xi} / 250.0) * ({v} - V_K)",
        ]

        # Calcium dynamics
        ca_dynamics_funcs = [
            lambda s, r, v: -self.model.phi
            * self.model.g_Ca
            * self.model.area
            * s**2
            * r
            * (v - self.model.V_Ca),
            lambda xi: -self.model.Beta * xi,
        ]
        ca_dynamics_names = [
            lambda s, r, v: f"-phi * g_Ca * area * {s}^2 * {r} * ({v} - V_Ca)",
            lambda xi: f"-Beta * {xi}",
        ]

        library_functions = gate_funcs + ion_current_funcs + ca_dynamics_funcs
        function_names = gate_names + ion_current_names + ca_dynamics_names

        return ps.CustomLibrary(
            library_functions=library_functions, function_names=function_names
        )

    def get_feature_lib(self):
        return self.create_base_functions() + ps.PolynomialLibrary(degree=1)
