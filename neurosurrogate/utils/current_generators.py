import math
import random

import numpy as np


def generate_steady(value: float):
    """一定の電流を生成する"""

    def apply(dset_i_ext: np.ndarray) -> None:
        dset_i_ext[:] = value

    return apply


def generate_rand_pulse(
    max_val: int = 20,
    pulse_step: int = 2000,
    flow_rate: float = 0.5,
    baseline: float = 0.0,
):
    def apply(dset_i_ext: np.ndarray) -> None:
        iteration = len(dset_i_ext)
        for n in range(math.floor(iteration / pulse_step)):
            v = random.randint(0, max_val) if random.random() < flow_rate else baseline
            dset_i_ext[n * pulse_step : (n + 1) * pulse_step] = v

    return apply


def generate_gauss_rand_pulse(
    max_val: int = 20,
    pulse_step: int = 2000,
    flow_rate: float = 0.5,
    mu: float = 0,
    sigma: float = 5,
    baseline: float = 0.0,
):
    def apply(dset_i_ext: np.ndarray) -> None:
        iteration = len(dset_i_ext)
        for n in range(math.floor(iteration / pulse_step)):
            if random.random() < flow_rate:
                v = np.clip(random.gauss(mu=mu, sigma=sigma), baseline, max_val)
            else:
                v = baseline
            dset_i_ext[n * pulse_step : (n + 1) * pulse_step] = v

    return apply


def generate_discretized(
    pulse_step: int = 2000,
    options: list = [-5, 6.2, 6.3, 5],
    weights: list = [1, 1, 1, 1],
    sigma: float = 0.1,
):
    def apply(dset_i_ext: np.ndarray) -> None:
        iteration = len(dset_i_ext)
        for n in range(math.floor(iteration / pulse_step)):
            chosen = random.choices(options, weights=weights, k=1)[0]
            chosen = chosen + random.gauss(mu=0, sigma=sigma)
            dset_i_ext[n * pulse_step : (n + 1) * pulse_step] = chosen

    return apply


def add_white_noise(sigma: float = 0.1):
    def apply(dset_i_ext: np.ndarray) -> None:
        dset_i_ext += np.random.normal(0, sigma, len(dset_i_ext))

    return apply
