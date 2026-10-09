"""波形/スパイクの指標計算。MLflow 非依存。

外へ出すのは `compare` 1 つ = 1 ペア (原系, 置換系) の全指標を素の値の dict で返す。
並べ方と描き方は持たない (図は live-textbook の冊が JSON を読んで描く)。発散判定
(`diverged`) は発散ログからも呼ばれる共通述語なので `core/diverge.py` に置く。
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import TypeVar

import efel
import numpy as np
import xarray as xr

from ..core import access
from ..core.diverge import diverged

T = TypeVar("T")
R = TypeVar("R")

_MEDIAN_FEATURES: list[str] = [
    # --- 電位の絶対値・相対値 [mV] ---
    "peak_voltage",  # 各 AP のピーク時の電位（V[peak_indices]）
    # AP 振幅: peak_voltage - V[AP_begin_indices]（spike onset からの相対高）
    "AP_amplitude",
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


@dataclass
class _DynamicMetrics:
    """電圧・eFEL特徴量を計算するデータ層。下記の純粋関数群から参照される
    (計算そのものはここで完結し、指標側は cached の値を読むだけ)。"""

    original: xr.Dataset = field(repr=False)
    surrogate: xr.Dataset = field(repr=False)
    comp_id: int
    dt: float

    @cached_property
    def voltages(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            access.potential(self.original, self.comp_id),
            access.potential(self.surrogate, self.comp_id),
        )

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
            # スパイクなし時に eFEL が RuntimeWarning、nan 変換で無害
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, module=r"efel\.*"
            )
            orig_feat, surr_feat = efel.get_feature_values(
                [_to_trace(orig_v), _to_trace(surr_v)],
                _EFEL_FEATURES,
            )
        return orig_feat, surr_feat

    @cached_property
    def peaks(self) -> tuple[list, list]:
        orig_feat, surr_feat = self.efel
        p, q = orig_feat.get("peak_indices"), surr_feat.get("peak_indices")
        return (list(p) if p is not None else []), (list(q) if q is not None else [])


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


def _diff_or_nan(o: float, s: float) -> float:
    """o - s。ただし片方でも nan なら nan を返す（差分計算の nan 伝播）。"""
    return o - s if not (np.isnan(o) or np.isnan(s)) else _NAN


def _corr_or_nan(a, b) -> float:
    """a, b の Pearson 相関。片方でも None なら nan。"""
    if a is None or b is None:
        return _NAN
    return float(np.corrcoef(a, b)[0, 1])


def _pair(fn: Callable[[T], R], pair: tuple[T, T]) -> tuple[R, R]:
    """(orig, surr) ペアに fn を適用して (fn(orig), fn(surr)) を返す。"""
    return fn(pair[0]), fn(pair[1])


# ---------------------------------------------------------------------------
# スパイク指標（純粋関数群、_DynamicMetrics を引数で受ける）
# ---------------------------------------------------------------------------


def _n_spikes(dm: _DynamicMetrics) -> tuple[int, int]:
    """(n_orig, n_surr): 各信号のスパイク数。"""
    return _pair(len, dm.peaks)


def _spike_shape_corr(dm: _DynamicMetrics) -> dict:
    """平均スパイクテンプレート間の Pearson 相関（1に近いほど形状が一致）。"""
    half_win = int(2.0 / dm.dt)

    def _mean_template(v, peaks):
        snippets = [
            v[p - half_win : p + half_win + 1]
            for p in peaks
            if p - half_win >= 0 and p + half_win + 1 <= len(v)
        ]
        return np.mean(snippets, axis=0) if snippets else None

    orig_tmpl, surr_tmpl = (
        _mean_template(v, p) for v, p in zip(dm.voltages, dm.peaks, strict=True)
    )
    return {"spike_shape_corr": _corr_or_nan(orig_tmpl, surr_tmpl)}


def _spike_features(dm: _DynamicMetrics) -> dict[str, tuple[list, list]]:
    """eFEL 特徴量ごとの (orig, surr) の全 AP 分。比べる AP は読む側が選ぶ。"""

    def values(features: dict, feat: str) -> list:
        found = features.get(feat)
        return [] if found is None else [float(x) for x in found]

    orig_feat, surr_feat = dm.efel
    return {
        feat: (values(orig_feat, feat), values(surr_feat, feat))
        for feat in _MEDIAN_FEATURES
    }


# ---------------------------------------------------------------------------
# 波形・発火パターン指標（純粋関数群）
# ---------------------------------------------------------------------------


def _waveform_error(dm: _DynamicMetrics) -> dict:
    """RMSE/MAE の波形誤差スカラー。"""
    orig_v, surr_v = dm.voltages
    return {
        "rmse": float(np.sqrt(np.mean((orig_v - surr_v) ** 2))),
        "mae": float(np.mean(np.abs(orig_v - surr_v))),
    }


def _latency(dm: _DynamicMetrics) -> tuple[float, float]:
    return _pair(lambda f: _at_or_nan(f.get("time_to_first_spike"), 0), dm.efel)


def _isi_stat(dm: _DynamicMetrics, fn) -> tuple[float, float]:
    isi = _pair(lambda f: f.get("ISI_values"), dm.efel)
    return _pair(lambda a: _or_nan(fn, a), isi)


def compare(
    original: xr.Dataset, surrogate: xr.Dataset, comp_id: int, dt: float
) -> dict:
    """1 ペアの comp `comp_id` の全指標。

    `rows` は原系と置換系の両方にある指標の (orig, surr)、`scalars` は両者の比較で
    決まる指標、`spikes` は AP ごとの特徴量。値は nan を含む素の float。`diverged` は
    置換系の電位が破綻したか (波形の図はそのとき置換系を描かない)。"""
    dm = _DynamicMetrics(original, surrogate, comp_id, dt)
    o_n, s_n = _n_spikes(dm)
    isi_mean = _isi_stat(dm, np.mean)
    return {
        "rows": {
            "spike_count": (float(o_n), float(s_n)),
            "latency": _latency(dm),
            "mean_isi": isi_mean,
            "std_isi": _isi_stat(dm, np.std),
        },
        "scalars": {
            **_waveform_error(dm),
            "periodicity_gap": abs(_diff_or_nan(*isi_mean)),
            **_spike_shape_corr(dm),
        },
        "spikes": _spike_features(dm),
        "diverged": diverged(dm.voltages[1]),
    }
