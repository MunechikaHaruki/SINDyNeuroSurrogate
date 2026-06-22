import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

import efel
import numpy as np
import pandas as pd

_MEDIAN_FEATURES: list[str] = [
    # --- 電位の絶対値・相対値 [mV] ---
    "peak_voltage",  # 各 AP のピーク時の電位（V[peak_indices]）
    "AP_amplitude",  # AP 振幅: peak_voltage - V[AP_begin_indices]（spike onset からの相対高）
    "AP_begin_voltage",  # spike start 時点の電位。spike start は dV/dt > 10 V/s が
    # 5 点以上続く最初の時点として定義される（実質的な閾値電位）
    # --- 電位変化速度 [V/s]（= mV/ms）---
    "AP_rise_rate",  # 立ち上がり相の平均変化速度:
    # (V[peak] - V[AP_begin]) / (T[peak] - T[AP_begin])
    "AP_fall_rate",  # 下降相の平均変化速度:
    # (V[AP_end] - V[peak]) / (T[AP_end] - T[peak])（負値）
    # --- 時間幅 [ms] ---
    "AP_duration_half_width",  # 半値全幅: 立ち上がり相と下降相で
    # (V[peak] - V[AP_begin]) / 2 に達する点の時間差
    "AP_rise_time",  # spike start からピークまでの所要時間: T[peak] - T[AP_begin]
    # （デフォルトでは振幅の 0%→100% 窓; rise_start_perc / rise_end_perc で変更可）
    "AP_fall_time",  # ピークから AP_end_indices までの所要時間: T[AP_end] - T[peak]
    # --- AHP（後過分極）---
    "AHP_depth",  # 1 番目の AHP の電位を voltage_base からの相対値で表現 [mV]:
    # min_AHP_values - voltage_base（通常は負値）
    "AHP_time_from_peak",  # AP ピークから最初の AHP minimum までの時間 [ms]:
    # T[min_AHP_indices] - T[peak_indices]
]
_EFEL_FEATURES = [
    "peak_indices",
    "ISI_values",
    "time_to_first_spike",
    *_MEDIAN_FEATURES,
]

_NAN = float("nan")


def _or_nan(fn, arr) -> float:
    """arr が None/空なら nan、それ以外は float(fn(arr))。"""
    if arr is None or len(arr) == 0:
        return _NAN
    return float(fn(arr))


def _at_or_nan(arr, idx: int) -> float:
    """arr[idx] を float で返す。arr が None/idx 範囲外なら nan。"""
    if arr is None or idx >= len(arr):
        return _NAN
    return float(arr[idx])


def _diff(o: float, s: float) -> float:
    """o - s。ただし片方でも nan なら nan を返す（差分計算の nan 伝播）。"""
    return o - s if not (np.isnan(o) or np.isnan(s)) else _NAN


def _corr_or_nan(a, b) -> float:
    """a, b の Pearson 相関。片方でも None なら nan。"""
    if a is None or b is None:
        return _NAN
    return float(np.corrcoef(a, b)[0, 1])


def _pair(fn, pair: tuple):
    """(orig, surr) ペアに fn を適用して (fn(orig), fn(surr)) を返す。"""
    return fn(pair[0]), fn(pair[1])


def _row(name: str, o: float, s: float, col: str = "metric") -> dict:
    """orig/surr/orig-surr の DataFrame 行 dict を生成。col で index 列名を指定。"""
    return {col: name, "orig": o, "surr": s, "orig-surr": _diff(o, s)}


@dataclass
class DynamicMetrics:
    """電圧・eFEL特徴量を計算するデータ層。下記の純粋関数群から参照される。"""

    original: Any = field(repr=False)
    surrogate: Any = field(repr=False)
    comp_id: int
    dt: float

    @cached_property
    def voltages(self) -> tuple[np.ndarray, np.ndarray]:
        orig_v = (
            self.original["vars"]
            .sel(gate=False, comp_id=self.comp_id)
            .to_numpy()
            .squeeze()
        )
        surr_v = (
            self.surrogate["vars"]
            .sel(gate=False, comp_id=self.comp_id)
            .to_numpy()
            .squeeze()
        )
        return orig_v, surr_v

    @cached_property
    def efel(self) -> tuple[dict, dict]:
        orig_v, surr_v = self.voltages

        def _to_trace(v: np.ndarray) -> dict:
            time = np.arange(len(v), dtype=float) * self.dt
            # stim_start を 1 サンプル後ろにずらして AHP baseline 計算に必要な区間を確保
            return {
                "T": time,
                "V": v.astype(float),
                "stim_start": [time[min(1, len(time) - 1)]],
                "stim_end": [time[-1]],
            }

        with warnings.catch_warnings():
            # スパイクなし時に eFEL が RuntimeWarning を出すが、nan に変換するため問題ない
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, module=r"efel\.*"
            )
            orig_feat, surr_feat = efel.get_feature_values(  # type: ignore[reportCallIssue]
                [_to_trace(orig_v), _to_trace(surr_v)],
                _EFEL_FEATURES,
            )
        return orig_feat, surr_feat

    @cached_property
    def peaks(self) -> tuple[list, list]:
        orig_feat, surr_feat = self.efel
        p, q = orig_feat.get("peak_indices"), surr_feat.get("peak_indices")
        return (list(p) if p is not None else []), (list(q) if q is not None else [])


# ---------------------------------------------------------------------------
# スパイク指標（純粋関数群、DynamicMetrics を引数で受ける）
# ---------------------------------------------------------------------------


def n_spikes(dm: DynamicMetrics) -> tuple[int, int]:
    """(n_orig, n_surr): 各信号のスパイク数。"""
    return _pair(len, dm.peaks)


def spike_shape_corr(dm: DynamicMetrics) -> dict:
    """平均スパイクテンプレート間の Pearson 相関（1に近いほど形状が一致）。"""
    half_win = int(2.0 / dm.dt)

    def _mean_template(v, peaks):
        snippets = [
            v[p - half_win : p + half_win + 1]
            for p in peaks
            if p - half_win >= 0 and p + half_win + 1 <= len(v)
        ]
        return np.mean(snippets, axis=0) if snippets else None

    orig_tmpl, surr_tmpl = (_mean_template(v, p) for v, p in zip(dm.voltages, dm.peaks))
    return {"spike_shape_corr": _corr_or_nan(orig_tmpl, surr_tmpl)}


def spike_features_df(
    dm: DynamicMetrics, spike: int | Literal["median"] = "median"
) -> pd.DataFrame:
    """median AP の eFEL 特徴量を orig/surr/orig-surr で並べた DataFrame。"""

    def _pick(arr) -> float:
        if spike == "median":
            return _or_nan(np.median, arr)
        return _at_or_nan(arr, spike)

    rows = [
        _row(feat, *_pair(lambda f: _pick(f.get(feat)), dm.efel), col="feature")
        for feat in _MEDIAN_FEATURES
    ]
    return pd.DataFrame(rows).set_index("feature")


# ---------------------------------------------------------------------------
# 波形・発火パターン指標（純粋関数群）
# ---------------------------------------------------------------------------


def _waveform_error(dm: DynamicMetrics) -> dict:
    """RMSE/MAE の波形誤差スカラー。"""
    orig_v, surr_v = dm.voltages
    return {
        "rmse": float(np.sqrt(np.mean((orig_v - surr_v) ** 2))),
        "mae": float(np.mean(np.abs(orig_v - surr_v))),
    }


def _spike_counts_df(dm: DynamicMetrics) -> pd.DataFrame:
    o, s = _pair(lambda p: float(len(p)), dm.peaks)
    return pd.DataFrame([_row("spike_count", o, s)]).set_index("metric")


def _timing_df(dm: DynamicMetrics) -> pd.DataFrame:
    o, s = _pair(lambda f: _at_or_nan(f.get("time_to_first_spike"), 0), dm.efel)
    return pd.DataFrame([_row("latency", o, s)]).set_index("metric")


def _isi_stats_df(dm: DynamicMetrics) -> pd.DataFrame:
    isi = _pair(lambda f: f.get("ISI_values"), dm.efel)
    o_mean, s_mean = _pair(lambda a: _or_nan(np.mean, a), isi)
    o_std, s_std = _pair(lambda a: _or_nan(np.std, a), isi)
    return pd.DataFrame(
        [_row("mean_isi", o_mean, s_mean), _row("std_isi", o_std, s_std)]
    ).set_index("metric")


def waveform_summary_df(dm: DynamicMetrics) -> pd.DataFrame:
    """spike_count / latency / mean_isi / std_isi を縦に並べた DataFrame。"""
    return pd.concat([_spike_counts_df(dm), _timing_df(dm), _isi_stats_df(dm)])


def waveform_summary(dm: DynamicMetrics) -> dict:
    """波形誤差 + サマリスカラー（spike_count_diff/latency_error/periodicity_gap）。"""
    df = waveform_summary_df(dm)
    return {
        **_waveform_error(dm),
        "orig_spike_count": int(df.loc["spike_count", "orig"]),
        "surr_spike_count": int(df.loc["spike_count", "surr"]),
        "spike_count_diff": abs(int(df.loc["spike_count", "orig-surr"])),
        "latency_error": abs(float(df.loc["latency", "orig-surr"])),
        "periodicity_gap": abs(float(df.loc["mean_isi", "orig-surr"])),
    }
