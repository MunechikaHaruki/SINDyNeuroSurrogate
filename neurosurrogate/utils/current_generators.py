import math
import random

import numpy as np

DEFAULT_ITER = 80000

CURRENT_GENERATER = {}


def current_decorator(func):
    """HHモデル用の関数ラッパー"""

    def wrapper(*args, **kwargs):
        # Set random seeds for reproducibility
        current_seed = kwargs.pop("current_seed", 0)
        random.seed(current_seed)
        np.random.seed(current_seed)
        iteration = kwargs.pop("iteration", DEFAULT_ITER)
        dset_i_ext = np.zeros(shape=(iteration,))
        func(dset_i_ext, iteration, *args, **kwargs)
        return dset_i_ext

    CURRENT_GENERATER[func.__name__] = wrapper

    return wrapper


@current_decorator
def hh_rand_pulse(
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
def hh_steady(dset_i_ext, iteration, value: float):
    """一定の電流を生成する"""
    dset_i_ext[:] = np.full(iteration, value)


@current_decorator
def hh_gauss_rand_pulse(
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
def hh_discretized(
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


@current_decorator
def hh_variable_width(
    dset_i_ext,
    iteration,
    max_val: int = 20,
    pulse_step: int = 2000,
    flow_rate: float = 0.5,
    mu=0,
    sigma=5,
):
    terminus = 0
    while terminus < iteration:
        pulse_step = random.randint(1, pulse_step)
        v = random.gauss(mu=mu, sigma=sigma) if random.random() < flow_rate else 0.0
        v = np.clip(v, 0, max_val)
        start = terminus
        terminus = min(terminus + pulse_step, iteration)
        dset_i_ext[start:terminus] = np.full(terminus - start, v)


# Traub用のコード
# def generate_traub_steady_current(
#     value: float, compartment_index: int
# ) -> Callable[[h5py.Dataset], None]:
# iter = dset_i_ext.shape[0]
# # dset_I_ext は (iter, NC) の形状なので、特定の列を更新
# dset_i_ext[:, compartment_index] = np.full(iter, value)
