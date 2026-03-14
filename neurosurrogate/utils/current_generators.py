import math
import random

import numpy as np


def current_decorator(func):
    """電流生成関数ラッパー"""

    def wrapper(*args, **kwargs):
        # Set random seeds for reproducibility
        current_seed = kwargs.pop("current_seed")
        random.seed(current_seed)
        np.random.seed(current_seed)
        iteration = kwargs.pop("iteration")
        silence_steps = kwargs.pop("silence_steps")
        dset_i_ext = np.zeros(shape=(iteration,))
        func(dset_i_ext, iteration, *args, **kwargs)

        dset_i_ext[:silence_steps] = 0  # 最初の電流を初期化
        return dset_i_ext

    return wrapper


@current_decorator
def generate_rand_pulse(
    dset_i_ext,
    iteration,
    max_val: int = 20,
    pulse_step: int = 2000,
    flow_rate: float = 0.5,
    baseline: float = 0.0,
):
    for n in range(math.floor(iteration / pulse_step)):
        v = random.randint(0, max_val) if random.random() < flow_rate else baseline
        dset_i_ext[n * pulse_step : (n + 1) * pulse_step] = np.full(pulse_step, v)


@current_decorator
def generate_steady(dset_i_ext, iteration, value: float):
    """一定の電流を生成する"""
    dset_i_ext[:] = np.full(iteration, value)


@current_decorator
def generate_gauss_rand_pulse(
    dset_i_ext,
    iteration,
    max_val: int = 20,
    pulse_step: int = 2000,
    flow_rate: float = 0.5,
    mu=0,
    sigma=5,
    baseline: float = 0.0,
):
    for n in range(math.floor(iteration / pulse_step)):
        if random.random() < flow_rate:
            v = random.gauss(mu=mu, sigma=sigma)
            v = np.clip(v, baseline, max_val)
        else:
            v = baseline
        dset_i_ext[n * pulse_step : (n + 1) * pulse_step] = np.full(pulse_step, v)


@current_decorator
def generate_discretized(
    dset_i_ext,
    iteration,
    pulse_step=2000,
    options=[-5, 6.2, 6.3, 5],
    weights=[1, 1, 1, 1],
    sigma=0.1,
):
    for n in range(math.floor(iteration / pulse_step)):
        chosen = random.choices(options, weights=weights, k=1)[0]
        chosen = chosen + random.gauss(mu=0, sigma=sigma)
        dset_i_ext[n * pulse_step : (n + 1) * pulse_step] = np.full(pulse_step, chosen)
