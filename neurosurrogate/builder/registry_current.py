import math

import numpy as np


def generate_steady(value: float):
    """一定の電流を生成する。value [μA/cm²]"""

    def apply(dset_i_ext: np.ndarray) -> None:
        dset_i_ext[:] = value

    return apply


def generate_rand_pulse(
    max_val: int = 20,
    pulse_step: int = 2000,
    flow_rate: float = 0.5,
    baseline: float = 0.0,
    seed: int = 0,
):
    """ランダムなパルス電流を生成する。max_val [μA/cm²]、pulse_step [steps]、baseline [μA/cm²]"""

    def apply(dset_i_ext: np.ndarray) -> None:
        rng = np.random.default_rng(seed)
        iteration = len(dset_i_ext)
        for n in range(math.floor(iteration / pulse_step)):
            v = rng.integers(0, max_val) if rng.random() < flow_rate else baseline
            dset_i_ext[n * pulse_step : (n + 1) * pulse_step] = v

    return apply


def generate_ramp(start: float, stop: float):
    """線形に増加・減少する電流を生成する。start/stop [μA/cm²]"""

    def apply(dset_i_ext: np.ndarray) -> None:
        dset_i_ext[:] = np.linspace(start, stop, len(dset_i_ext))

    return apply


def generate_step(values: list, step_duration: int):
    """段階的に変化する電流を生成する。values [μA/cm²]、step_duration [steps]"""

    def apply(dset_i_ext: np.ndarray) -> None:
        iteration = len(dset_i_ext)
        for i, value in enumerate(values):
            start = i * step_duration
            end = min((i + 1) * step_duration, iteration)
            if start >= iteration:
                break
            dset_i_ext[start:end] = value

    return apply


def generate_sinusoidal(amplitude: float = 7.5, frequency: float = 10.0, baseline: float = 7.5, dt: float = 0.01):
    """サイン波電流を生成する。baseline ± amplitude で振動。amplitude/baseline [μA/cm²]、frequency [Hz]、dt [ms/step]"""

    def apply(dset_i_ext: np.ndarray) -> None:
        iteration = len(dset_i_ext)
        t = np.arange(iteration) * dt * 1e-3  # ms → s
        dset_i_ext[:] = baseline + amplitude * np.sin(2 * np.pi * frequency * t)

    return apply


def generate_chirp(
    amplitude: float = 7.5,
    f_start: float = 1.0,
    f_stop: float = 100.0,
    dt: float = 0.01,
    baseline: float = 7.5,
):
    """周波数が時間とともに変化するサイン波電流を生成する。amplitude[μA/cm²]、f_start/f_stop[Hz]、dt[ms/step]"""

    def apply(dset_i_ext: np.ndarray) -> None:
        iteration = len(dset_i_ext)
        t = np.arange(iteration) * dt * 1e-3  # ms → s
        total_time = iteration * dt * 1e-3
        # 周波数を線形にスイープ
        phase = 2 * np.pi * (f_start * t + (f_stop - f_start) * t**2 / (2 * total_time))
        dset_i_ext[:] = baseline + amplitude * np.sin(phase)

    return apply


def generate_discretized(
    pulse_step: int = 2000,
    options: list = [-5, 6.2, 6.3, 5],
    weights: list = [1, 1, 1, 1],
    sigma: float = 0.1,
    seed: int = 0,
):
    """離散値からランダムに選んだパルス電流を生成する。options [μA/cm²]、pulse_step [steps]、sigma [μA/cm²]"""

    def apply(dset_i_ext: np.ndarray) -> None:
        rng = np.random.default_rng(seed)
        p = np.array(weights) / sum(weights)
        iteration = len(dset_i_ext)
        for n in range(math.floor(iteration / pulse_step)):
            chosen = rng.choice(options, p=p) + rng.normal(0, sigma)
            dset_i_ext[n * pulse_step : (n + 1) * pulse_step] = chosen

    return apply


def add_white_noise(sigma: float = 0.1):
    """既存の電流にガウスホワイトノイズを加算する。sigma [μA/cm²]"""

    def apply(dset_i_ext: np.ndarray) -> None:
        dset_i_ext += np.random.normal(0, sigma, len(dset_i_ext))

    return apply


FUNC_MAP = {
    "steady": generate_steady,
    "random": generate_rand_pulse,
    "ramp": generate_ramp,
    "step": generate_step,
    "sinousoidal": generate_sinusoidal,
    "chirp": generate_chirp,
    "discretized": generate_discretized,
    "noise": add_white_noise,
}
