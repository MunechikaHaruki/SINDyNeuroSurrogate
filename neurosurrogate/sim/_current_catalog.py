"""**注入電流の名前 → 波形の対応表** (`SimSpec.current_type` が引く選択肢)。

`spec` が仕様を実体化するときだけ引く sim 内部の表なので `_` 付き。適用先モデルの
対応表は `neurons` が持つ (組み方と同じ語彙に置く)。
"""

import functools
import inspect
import math
from collections.abc import Callable
from typing import Literal

import numpy as np


def _current_generator(fn: Callable) -> Callable:
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
    wrapper.__signature__ = inspect.signature(fn).replace(  # type: ignore[attr-defined]
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


@_current_generator
def _generate_steady(value: float = 10):
    """一定の電流を生成する。value [μA/cm²]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        active[:] = value

    return apply


@_current_generator
def _generate_ramp(amplitude: float = 30, direction: Literal["up", "down"] = "up"):
    """線形に増加・減少する電流を生成する。amplitude [μA/cm²]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        lo, hi = (amplitude, 0.0) if direction == "down" else (0.0, amplitude)
        active[:] = np.linspace(lo, hi, len(active))

    return apply


_LINEAR_FUNC: dict[str, Callable[..., Callable[[float], np.ndarray]]] = {
    "lin&steady": _generate_steady,
    "lin&ramp": _generate_ramp,
}


# ---------------------------------------------------------------------------
# 周期性の電流
# ---------------------------------------------------------------------------


@_current_generator
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


@_current_generator
def _generate_chirp(
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


@_current_generator
def _generate_pulse_train(
    amplitude: float = 20.0,
    frequency: float = 20.0,
    baseline: float = 0.0,
):
    """周期的な矩形パルス列。**流れる幅 = 流れない幅** (1 周期の半分ずつ = duty 50%)
    なので波形の自由度は frequency だけ = 掃引軸に取ると進むほどパルスが詰まる。
    amplitude/baseline [μA/cm²]、frequency [Hz]"""

    def apply(active: np.ndarray, dt: float) -> None:
        half = 500.0 / frequency  # 周期 (1000/f [ms]) の半分 = ON 幅 = OFF 幅
        active[:] = np.where(
            (np.arange(len(active)) * dt) % (2 * half) < half, amplitude, baseline
        )

    return apply


_PERIODIC_FUNC: dict[str, Callable[..., Callable[[float], np.ndarray]]] = {
    "periodic&sinousoidal": _generate_sinousoidal,
    "periodic&chirp": _generate_chirp,
    "periodic&pulse": _generate_pulse_train,
}


# ---------------------------------------------------------------------------
# ランダムな電流
# ---------------------------------------------------------------------------


@_current_generator
def _generate_rand_pulse(
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


@_current_generator
def _generate_discretized(
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


@_current_generator
def _generate_poisson_synapse(
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


_RANDOM_FUNC: dict[str, Callable[..., Callable[[float], np.ndarray]]] = {
    "random": _generate_rand_pulse,
    "random&discretized": _generate_discretized,
    "random&poisson_synapse": _generate_poisson_synapse,
}


# ---------------------------------------------------------------------------
# Others
# ---------------------------------------------------------------------------


@_current_generator
def _generate_step(values: list, step_duration: int):
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


@_current_generator
def _add_white_noise(sigma: float = 0.1):
    """既存の電流にガウスホワイトノイズを加算する。sigma [μA/cm²]"""

    def apply(active: np.ndarray, _dt: float) -> None:
        active += np.random.normal(0, sigma, len(active))

    return apply


def train(duration: float = 9000, seed: int = 991927697):
    """学習時電流。波形パラメータ固定 (duration/seed のみ可変)。
    seed = 離散値パルス列の乱数実現 (同分布の別サンプルで頑健性を見る)。"""
    return _generate_discretized(
        options=[-5, 1.3, 6.3, 20],
        weights=[0.3, 1, 1, 1],
        sigma=1,
        seed=seed,
        silence_duration=80,
        duration=duration,
    )


def _traub_soma_dc(value: float = 1e-4 / 3.320e-5):
    """traub.c の soma DC 注入を再現。C は i_inj[soma]=1e-4[μA]/area[soma] を全時刻
    一定注入 (silence 無し, T=200ms)。MC 規約では注入も密度 [μA/cm²] なので
    builder 値がそのまま流入する。
    既定値=1e-4/area[soma] (area[soma]=3.320e-5 [cm²], traub19 SOMA_IDX)。"""
    return _generate_steady(value, silence_duration=0, duration=200)


_OTHER_FUNC: dict[str, Callable[..., Callable[[float], np.ndarray]]] = {
    "train": train,
    "_traub_soma_dc": _traub_soma_dc,
    "step": _generate_step,
    "noise": _add_white_noise,
}


CURRENT_MAP: dict[str, Callable[..., Callable[[float], np.ndarray]]] = {
    **_OTHER_FUNC,
    **_LINEAR_FUNC,
    **_RANDOM_FUNC,
    **_PERIODIC_FUNC,
}


PARAM_UNITS: dict[str, str] = {
    "value": "μA/cm²",
    "amplitude": "μA/cm²",
    "baseline": "μA/cm²",
    "max_val": "μA/cm²",
    "sigma": "μA/cm²",
    "frequency": "Hz",
    "f_start": "Hz",
    "f_stop": "Hz",
    "rate": "Hz",
}
