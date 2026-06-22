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


def _diff(o: float, s: float) -> float:
    """o - s。ただし片方でも nan なら nan を返す（差分計算の nan 伝播）。"""
    return o - s if not (np.isnan(o) or np.isnan(s)) else _NAN


@dataclass
class DynamicMetrics:
    """電圧・eFEL特徴量を計算するデータ層。WaveformMetrics/SpikeMetrics に共有される。"""

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

    @cached_property
    def mean_template(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        def _mean_template(v, peaks, half_win):
            snippets = [
                v[p - half_win : p + half_win + 1]
                for p in peaks
                if p - half_win >= 0 and p + half_win + 1 <= len(v)
            ]
            return np.mean(snippets, axis=0) if snippets else None

        orig_v, surr_v = self.voltages
        orig_peaks, surr_peaks = self.peaks
        half_win = int(2.0 / self.dt)
        orig_tmpl = _mean_template(orig_v, orig_peaks, half_win)
        surr_tmpl = _mean_template(surr_v, surr_peaks, half_win)
        return orig_tmpl, surr_tmpl


@dataclass
class SpikeMetrics:
    """スパイク形状指標（median AP誤差、spike_shape_corr）を計算する。"""

    _dm: DynamicMetrics = field(repr=False)

    @cached_property
    def _spike_shape_corr(self) -> dict:
        """平均スパイクテンプレート間の Pearson 相関（1に近いほど形状が一致）。"""
        orig_tmpl, surr_tmpl = self._dm.mean_template
        if orig_tmpl is None or surr_tmpl is None:
            return {"spike_shape_corr": _NAN}
        return {"spike_shape_corr": float(np.corrcoef(orig_tmpl, surr_tmpl)[0, 1])}

    @cached_property
    def n_spikes(self) -> tuple[int, int]:
        """(n_orig, n_surr): 各信号のスパイク数。"""
        orig_peaks, surr_peaks = self._dm.peaks
        return len(orig_peaks), len(surr_peaks)

    def compute(self) -> dict:
        return self._spike_shape_corr

    def to_df(self, spike: int | Literal["median"] = "median") -> pd.DataFrame:
        orig_feat, surr_feat = self._dm.efel

        def _pick(arr, idx: int | Literal["median"]) -> float:
            if arr is None or len(arr) == 0:
                return _NAN
            if idx == "median":
                return float(np.median(arr))
            return float(arr[idx]) if idx < len(arr) else _NAN

        rows = []
        for feat in _MEDIAN_FEATURES:
            o = _pick(orig_feat.get(feat), spike)
            s = _pick(surr_feat.get(feat), spike)
            rows.append(
                {"feature": feat, "orig": o, "surr": s, "orig-surr": _diff(o, s)}
            )
        return pd.DataFrame(rows).set_index("feature")


@dataclass
class WaveformMetrics:
    """波形・発火パターン指標（RMSE/MAE・スパイク数・タイミング・ISI）を計算する。"""

    _dm: DynamicMetrics = field(repr=False)

    @cached_property
    def _waveform_error(self) -> dict:
        orig_v, surr_v = self._dm.voltages
        return {
            "rmse": float(np.sqrt(np.mean((orig_v - surr_v) ** 2))),
            "mae": float(np.mean(np.abs(orig_v - surr_v))),
        }

    @cached_property
    def _spike_counts(self) -> pd.DataFrame:
        orig_peaks, surr_peaks = self._dm.peaks
        o, s = float(len(orig_peaks)), float(len(surr_peaks))
        return pd.DataFrame(
            [
                {"metric": "spike_count", "orig": o, "surr": s, "orig-surr": o - s},
            ]
        ).set_index("metric")

    @cached_property
    def _timing(self) -> pd.DataFrame:
        orig_feat, surr_feat = self._dm.efel
        orig_tfs = orig_feat.get("time_to_first_spike")
        surr_tfs = surr_feat.get("time_to_first_spike")
        o = float(orig_tfs[0]) if orig_tfs is not None else _NAN
        s = float(surr_tfs[0]) if surr_tfs is not None else _NAN
        return pd.DataFrame(
            [
                {"metric": "latency", "orig": o, "surr": s, "orig-surr": _diff(o, s)},
            ]
        ).set_index("metric")

    @cached_property
    def _isi_stats(self) -> pd.DataFrame:
        orig_feat, surr_feat = self._dm.efel
        orig_isi = orig_feat.get("ISI_values")
        surr_isi = surr_feat.get("ISI_values")
        o_mean, s_mean = _or_nan(np.mean, orig_isi), _or_nan(np.mean, surr_isi)
        o_std, s_std = _or_nan(np.std, orig_isi), _or_nan(np.std, surr_isi)
        return pd.DataFrame(
            [
                {
                    "metric": "mean_isi",
                    "orig": o_mean,
                    "surr": s_mean,
                    "orig-surr": _diff(o_mean, s_mean),
                },
                {
                    "metric": "std_isi",
                    "orig": o_std,
                    "surr": s_std,
                    "orig-surr": _diff(o_std, s_std),
                },
            ]
        ).set_index("metric")

    def to_df(self) -> pd.DataFrame:
        return pd.concat([self._spike_counts, self._timing, self._isi_stats])

    def compute(self) -> dict:
        df = self.to_df()
        return {
            **self._waveform_error,
            "orig_spike_count": int(df.loc["spike_count", "orig"]),
            "surr_spike_count": int(df.loc["spike_count", "surr"]),
            "spike_count_diff": abs(int(df.loc["spike_count", "orig-surr"])),
            "latency_error": abs(float(df.loc["latency", "orig-surr"])),
            "periodicity_gap": abs(float(df.loc["mean_isi", "orig-surr"])),
        }
