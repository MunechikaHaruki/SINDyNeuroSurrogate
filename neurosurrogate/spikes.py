"""1 本の波形のスパイクの特徴量 (eFEL)。元のモデルとの比較は図 (evaluation/) が持つ。"""

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
