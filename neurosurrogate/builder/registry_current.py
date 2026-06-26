import functools
import inspect
import math
from collections.abc import Callable

import numpy as np


def current_generator(fn: Callable) -> Callable:
    """silence_duration/duration を引数に持たない apply(active, dt) 返し関数を build(dt) 返し関数に昇格。"""

    @functools.wraps(fn)
    def wrapper(*args, silence_duration: float = 0.0, duration: float = 100.0, **kwargs):
        apply = fn(*args, **kwargs)

        def build(dt: float) -> np.ndarray:
            iteration = int(duration / dt)
            silence_steps = int(silence_duration / dt)
            if iteration - silence_steps <= silence_steps:
                raise ValueError(
                    f"silence_duration={silence_duration} が大きすぎます（duration={duration}）"
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


@current_generator
def generate_steady(value: float = 10):
    """一定の電流を生成する。value [μA/cm²]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        active[:] = value

    return apply


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
def generate_ramp(start: float, stop: float):
    """線形に増加・減少する電流を生成する。start/stop [μA/cm²]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        active[:] = np.linspace(start, stop, len(active))

    return apply


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
def generate_sinousoidal(
    amplitude: float = 7.5,
    frequency: float = 10.0,
    baseline: float = 7.5,
):
    """サイン波電流を生成する。baseline ± amplitude で振動。amplitude/baseline [μA/cm²]、frequency [Hz]"""

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


@current_generator
def generate_discretized(
    pulse_step: int = 2000,
    options: list = [-5, 6.2, 6.3, 5],  # noqa: B006
    weights: list = [1, 1, 1, 1],  # noqa: B006
    sigma: float = 0.1,
    seed: int = 0,
):
    """離散値からランダムに選んだパルス電流を生成する。options [μA/cm²]、pulse_step [steps]、sigma [μA/cm²]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        rng = np.random.default_rng(seed)
        p = np.array(weights) / sum(weights)
        n_active = len(active)
        for n in range(math.floor(n_active / pulse_step)):
            chosen = rng.choice(options, p=p) + rng.normal(0, sigma)
            active[n * pulse_step : (n + 1) * pulse_step] = chosen

    return apply


@current_generator
def add_white_noise(sigma: float = 0.1):
    """既存の電流にガウスホワイトノイズを加算する。sigma [μA/cm²]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        active += np.random.normal(0, sigma, len(active))

    return apply


FUNC_MAP = {
    "steady": generate_steady,
    "random": generate_rand_pulse,
    "ramp": generate_ramp,
    "step": generate_step,
    "sinousoidal": generate_sinousoidal,
    "chirp": generate_chirp,
    "discretized": generate_discretized,
    "noise": add_white_noise,
}
