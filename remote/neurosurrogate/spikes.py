"""波形のスパイクの特徴量 (eFEL)。run には書かず、API が頼まれるたびに呼ぶ。

`python -m neurosurrogate.spikes '[{"dt": 刻み, "n": 長さ}, ...]'` が、標準入力の
little-endian float64 を並びの長さで切った波形ごとに、特徴量を JSON の並びで書く
(nan は null)。run の形式は知らない。波形を読んで渡すのも、元のモデルとの比べも
API (../read/comparison.ts) が持つ。
"""

import json
import math
import sys
import warnings

import efel
import numpy as np

# AP ごとの特徴量。定義は eFEL の文書 (AP_begin は dV/dt > 10 V/s が 5 点続く所)。
AP_FEATURES = [
    "peak_voltage",
    "AP_amplitude",
    "AP_begin_voltage",
    "AP_rise_rate",
    "AP_fall_rate",
    "AP_duration_half_width",
    "AP_rise_time",
    "AP_fall_time",
    "AHP_depth",
    "AHP_time_from_peak",
]
FEATURES = ["peak_indices", "ISI_values", "time_to_first_spike", *AP_FEATURES]


def spike_features(v: np.ndarray, dt: float) -> dict[str, list[float]]:
    """特徴量の名前 → 値の並び。スパイクが無ければ空。"""
    time = np.arange(len(v), dtype=float) * dt
    # stim_start を 1 サンプル後ろにずらして AHP の基線の区間を確保する
    trace = {
        "T": time,
        "V": v.astype(float),
        "stim_start": [time[min(1, len(time) - 1)]],
        "stim_end": [time[-1]],
    }
    with warnings.catch_warnings():
        # スパイクが無いと eFEL が RuntimeWarning を出す。値は None になり空にする
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"efel\.*")
        (found,) = efel.get_feature_values([trace], FEATURES)
    return {
        name: [] if found[name] is None else [float(x) for x in found[name]]
        for name in FEATURES
    }


def _nulled(features: dict[str, list[float]]) -> dict[str, list[float | None]]:
    """JSON に書けない nan・無限大を null にする。"""
    return {
        name: [x if math.isfinite(x) else None for x in values]
        for name, values in features.items()
    }


def main(header: str) -> None:
    """標準入力の波形を header の長さで切り、波形ごとの特徴量を標準出力へ書く。"""
    traces = json.loads(header)
    values = np.frombuffer(sys.stdin.buffer.read(), dtype="<f8")
    if values.size != sum(t["n"] for t in traces):
        raise ValueError(f"波形の長さの和 {values.size} が header と合わない")
    found, start = [], 0
    for t in traces:
        v = values[start : start + t["n"]]
        found.append(_nulled(spike_features(v, float(t["dt"]))))
        start += t["n"]
    json.dump(found, sys.stdout)


if __name__ == "__main__":
    main(sys.argv[1])
