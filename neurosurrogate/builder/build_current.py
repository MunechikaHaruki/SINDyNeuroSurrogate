import math

import hydra
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
    seed: int | None = None,
):
    def apply(dset_i_ext: np.ndarray) -> None:
        rng = np.random.default_rng(seed)
        iteration = len(dset_i_ext)
        for n in range(math.floor(iteration / pulse_step)):
            v = rng.integers(0, max_val) if rng.random() < flow_rate else baseline
            dset_i_ext[n * pulse_step : (n + 1) * pulse_step] = v

    return apply


def generate_ramp(start: float, stop: float):
    """線形に増加・減少する電流を生成する"""

    def apply(dset_i_ext: np.ndarray) -> None:
        dset_i_ext[:] = np.linspace(start, stop, len(dset_i_ext))

    return apply


def generate_step(values: list, step_duration: int):
    """段階的に変化する電流を生成する"""

    def apply(dset_i_ext: np.ndarray) -> None:
        iteration = len(dset_i_ext)
        for i, value in enumerate(values):
            start = i * step_duration
            end = min((i + 1) * step_duration, iteration)
            if start >= iteration:
                break
            dset_i_ext[start:end] = value

    return apply


thisfile = "neurosurrogate.builder.build_current"

PIPE_FUNCS = {
    "steady": lambda **kw: {"_target_": f"{thisfile}.generate_steady", **kw},
    "random": lambda **kw: {"_target_": f"{thisfile}.generate_rand_pulse", **kw},
    "ramp": lambda **kw: {"_target_": f"{thisfile}.generate_ramp", **kw},
    "step": lambda **kw: {"_target_": f"{thisfile}.generate_step", **kw},
}

FUNC_MAP = {
    "steady": generate_steady,
    "random": generate_rand_pulse,
    "ramp": generate_ramp,
    "step": generate_step,
}


def build_current_pipeline(current_cfg):
    iteration = current_cfg["iteration"]
    silence_steps = current_cfg["silence_steps"]
    dset_i_ext = np.zeros(iteration)

    func = hydra.utils.instantiate(current_cfg["pipeline"])
    func(dset_i_ext)

    dset_i_ext[:silence_steps] = 0
    dset_i_ext[-silence_steps:] = 0
    return dset_i_ext


def generate_discretized(
    pulse_step: int = 2000,
    options: list = [-5, 6.2, 6.3, 5],
    weights: list = [1, 1, 1, 1],
    sigma: float = 0.1,
    seed: int | None = None,
):
    def apply(dset_i_ext: np.ndarray) -> None:
        rng = np.random.default_rng(seed)
        p = np.array(weights) / sum(weights)
        iteration = len(dset_i_ext)
        for n in range(math.floor(iteration / pulse_step)):
            chosen = rng.choice(options, p=p) + rng.normal(0, sigma)
            dset_i_ext[n * pulse_step : (n + 1) * pulse_step] = chosen

    return apply


def add_white_noise(sigma: float = 0.1):
    def apply(dset_i_ext: np.ndarray) -> None:
        dset_i_ext += np.random.normal(0, sigma, len(dset_i_ext))

    return apply


# テスト用の電流


def generate_sinusoidal(amplitude: float, frequency: float):
    """サイン波電流を生成する　frequencyの単位はHz、dtは秒"""

    def apply(dset_i_ext: np.ndarray) -> None:
        iteration = len(dset_i_ext)
        t = np.arange(iteration)
        dset_i_ext[:] = amplitude * np.sin(2 * np.pi * frequency * t)

    return apply


def generate_chirp(
    amplitude: float, f_start: float, f_stop: float, baseline: float = 0.0
):
    """周波数が時間とともに変化するサイン波電流を生成する"""

    def apply(dset_i_ext: np.ndarray) -> None:
        iteration = len(dset_i_ext)
        t = np.arange(iteration)
        # 周波数を線形にスイープ
        phase = 2 * np.pi * (f_start * t + (f_stop - f_start) * t**2 / (2 * iteration))
        dset_i_ext[:] = baseline + amplitude * np.sin(phase)

    return apply
