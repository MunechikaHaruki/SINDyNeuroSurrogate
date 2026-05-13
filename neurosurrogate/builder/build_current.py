import math
from dataclasses import dataclass

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
    seed: int = 0,
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


def generate_discretized(
    pulse_step: int = 2000,
    options: list = [-5, 6.2, 6.3, 5],
    weights: list = [1, 1, 1, 1],
    sigma: float = 0.1,
    seed: int = 0,
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


@dataclass
class CurrentConfig:
    iteration: int
    silence_steps: int
    pipeline: dict

    def build(self):
        dset_i_ext = np.zeros(self.iteration)

        if (
            self.iteration - self.silence_steps <= self.silence_steps
        ):  # active_end <=active_start
            raise ValueError(
                f"silence_steps={self.silence_steps} が大きすぎます（iteration={self.iteration}）"
            )
        active = dset_i_ext[self.silence_steps : self.iteration - self.silence_steps]
        func = hydra.utils.instantiate(self.pipeline)
        func(active)
        return dset_i_ext

    @staticmethod
    def build_pipeline(current_type: str, kw: dict) -> dict:
        return {
            "_target_": f"neurosurrogate.builder.build_current.{FUNC_MAP[current_type].__name__}",
            **kw,
        }

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "silence_steps": self.silence_steps,
            "pipeline": self.pipeline,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CurrentConfig":
        return cls(
            iteration=d["iteration"],
            silence_steps=d["silence_steps"],
            pipeline=d["pipeline"],
        )
