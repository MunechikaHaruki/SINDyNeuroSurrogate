import functools
import inspect
import math
from collections.abc import Callable
from typing import Literal

import numpy as np


def current_generator(fn: Callable) -> Callable:
    """silence_duration/duration を引数に持たない apply(active, dt) 返し関数を
    build(dt) 返し関数に昇格。"""

    @functools.wraps(fn)
    def wrapper(
        *args, silence_duration: float = 10.0, duration: float = 120.0, **kwargs
    ):
        apply = fn(*args, **kwargs)

        def build(dt: float) -> np.ndarray:
            iteration = int(duration / dt)
            silence_steps = int(silence_duration / dt)
            if iteration - silence_steps <= silence_steps:
                raise ValueError(
                    f"silence_duration={silence_duration} is too long"
                    f" (duration={duration})"
                )
            dset_i_ext = np.zeros(iteration)
            apply(dset_i_ext[silence_steps : iteration - silence_steps], dt)
            return dset_i_ext

        return build

    orig_params = list(inspect.signature(fn).parameters.values())
    wrapper.__signature__ = inspect.signature(fn).replace(
        parameters=orig_params
        + [
            inspect.Parameter(
                "silence_duration",
                inspect.Parameter.KEYWORD_ONLY,
                default=0.0,
                annotation=float,
            ),
            inspect.Parameter(
                "duration",
                inspect.Parameter.KEYWORD_ONLY,
                default=100.0,
                annotation=float,
            ),
        ]
    )
    return wrapper


# ---------------------------------------------------------------------------
# 線形の電流
# ---------------------------------------------------------------------------


@current_generator
def _generate_steady(value: float = 10):
    """一定の電流を生成する。value [μA/cm²]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        active[:] = value

    return apply


@current_generator
def _generate_ramp(amplitude: float = 30, direction: Literal["up", "down"] = "up"):
    """線形に増加・減少する電流を生成する。amplitude [μA/cm²]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        lo, hi = (amplitude, 0.0) if direction == "down" else (0.0, amplitude)
        active[:] = np.linspace(lo, hi, len(active))

    return apply


def steady(value: float = 10):
    return _generate_steady(value, silence_duration=10, duration=120)


def single_pulse(value: float = 10):
    return _generate_steady(value, silence_duration=10, duration=30)


def ramp(amplitude: float = 20, direction: Literal["up", "down"] = "up"):
    return _generate_ramp(amplitude, direction, silence_duration=0, duration=100)


LINEAR_FUNC = {
    "lin&steady": steady,
    "lin&steady&pulse": single_pulse,
    "lin&ramp": ramp,
}


# ---------------------------------------------------------------------------
# 周期性の電流
# ---------------------------------------------------------------------------


@current_generator
def _generate_sinousoidal(
    amplitude: float = 7.5,
    frequency: float = 10.0,
    baseline: float = 7.5,
):
    """サイン波電流を生成する。baseline ± amplitude で振動。amplitude/baseline [μA/cm²]
    frequency [Hz]"""

    def apply(active: np.ndarray, dt: float) -> None:
        t = np.arange(len(active)) * dt * 1e-3  # ms → s
        active[:] = baseline + amplitude * np.sin(2 * np.pi * frequency * t)

    return apply


@current_generator
def generate_chirp(
    amplitude: float = 7.5,
    f_start: float = 1.0,
    f_stop: float = 100.0,
    baseline: float = 7.5,
):
    """周波数が時間とともに変化するサイン波電流を生成する。amplitude[μA/cm²]、f_start/f_stop[Hz]"""

    def apply(active: np.ndarray, dt: float) -> None:
        n_active = len(active)
        t = np.arange(n_active) * dt * 1e-3  # ms → s
        total_time = n_active * dt * 1e-3
        phase = 2 * np.pi * (f_start * t + (f_stop - f_start) * t**2 / (2 * total_time))
        active[:] = baseline + amplitude * np.sin(phase)

    return apply


def sinousoidal(frequency: float = 50):
    return _generate_sinousoidal(
        amplitude=7.5,
        frequency=frequency,
        baseline=7.5,
        silence_duration=10,
        duration=520,
    )


PERIODIC_FUNC = {
    "periodic&sinousoidal": sinousoidal,
    "periodic&chirp": generate_chirp,
}


# ---------------------------------------------------------------------------
# ランダムな電流
# ---------------------------------------------------------------------------


@current_generator
def generate_rand_pulse(
    max_val: int = 20,
    pulse_step: int = 2000,
    flow_rate: float = 0.5,
    baseline: float = 0.0,
    seed: int = 0,
):
    """ランダムなパルス電流を生成する。max_val [μA/cm²]、pulse_step [steps]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        rng = np.random.default_rng(seed)
        n_active = len(active)
        for n in range(math.floor(n_active / pulse_step)):
            v = rng.integers(0, max_val) if rng.random() < flow_rate else baseline
            active[n * pulse_step : (n + 1) * pulse_step] = v

    return apply


@current_generator
def generate_discretized(
    pulse_step: int = 2000,
    options: list = [-5, 6.2, 6.3, 5],  # noqa: B006
    weights: list = [1, 1, 1, 1],  # noqa: B006
    sigma: float = 0.1,
    seed: int = 0,
):
    """離散値からランダムに選んだパルス電流を生成する。
    options [μA/cm²]、pulse_step [steps]、sigma [μA/cm²]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        rng = np.random.default_rng(seed)
        p = np.array(weights) / sum(weights)
        n_active = len(active)
        for n in range(math.floor(n_active / pulse_step)):
            chosen = rng.choice(options, p=p) + rng.normal(0, sigma)
            active[n * pulse_step : (n + 1) * pulse_step] = chosen

    return apply


@current_generator
def generate_poisson_synapse(
    rate: float = 20.0,
    amplitude: float = 20.0,
    tau_rise: float = 0.5,
    tau_decay: float = 5.0,
    seed: int = 0,
):
    """Poisson過程スパイク列 由来 シナプス電流を生成。
    2重指数 (α-like) カーネルで rise/decay。
    rate [Hz]、amplitude [μA/cm²] (単一スパイクピーク)、tau_rise/tau_decay [ms]"""

    def apply(active: np.ndarray, dt: float) -> None:
        rng = np.random.default_rng(seed)
        n_active = len(active)
        prob_per_step = rate * dt * 1e-3
        spikes = (rng.random(n_active) < prob_per_step).astype(np.float64)

        kernel_len = max(2, int(5 * tau_decay / dt))
        t_k = np.arange(kernel_len) * dt
        kernel = np.exp(-t_k / tau_decay) - np.exp(-t_k / tau_rise)
        kernel /= kernel.max()  # peak = 1
        kernel *= amplitude

        active[:] += np.convolve(spikes, kernel, mode="full")[:n_active]

    return apply


RANDOM_FUNC = {
    "random": generate_rand_pulse,
    "random&discretized": generate_discretized,
    "random&poisson_synapse": generate_poisson_synapse,
}


# ---------------------------------------------------------------------------
# Others
# ---------------------------------------------------------------------------


@current_generator
def generate_step(values: list, step_duration: int):
    """段階的に変化する電流を生成する。values [μA/cm²]、step_duration [steps]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        n_active = len(active)
        for i, value in enumerate(values):
            start = i * step_duration
            end = min((i + 1) * step_duration, n_active)
            if start >= n_active:
                break
            active[start:end] = value

    return apply


@current_generator
def add_white_noise(sigma: float = 0.1):
    """既存の電流にガウスホワイトノイズを加算する。sigma [μA/cm²]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        active += np.random.normal(0, sigma, len(active))

    return apply


def train():
    """学習時電流。パラメータ完全固定。"""
    return generate_discretized(
        options=[-5, 1.3, 6.3, 20],
        weights=[0.3, 1, 1, 1],
        sigma=1,
        seed=991927697,
        silence_duration=80,
        duration=9000,
    )


OTHER_FUNC = {
    "train": train,
    "step": generate_step,
    "noise": add_white_noise,
}


FUNC_MAP = {
    **OTHER_FUNC,
    **LINEAR_FUNC,
    **RANDOM_FUNC,
    **PERIODIC_FUNC,
}
