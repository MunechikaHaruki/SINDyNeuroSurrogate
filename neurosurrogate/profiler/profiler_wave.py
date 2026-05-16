import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import efel
import numpy as np
import pandas as pd

_MEDIAN_FEATURES: list[str] = [
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

_EFEL_FEATURES = [
    "peak_indices",
    "ISI_values",
    "time_to_first_spike",
    *_MEDIAN_FEATURES,
]


def _or_nan(fn, arr) -> float:
    if arr is None or len(arr) == 0:
        return float("nan")
    return float(fn(arr))


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
        half_win = 50
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
            return {"spike_shape_corr": float("nan")}
        return {"spike_shape_corr": float(np.corrcoef(orig_tmpl, surr_tmpl)[0, 1])}

    def compute(self) -> dict:
        return self._spike_shape_corr

    def to_df(self) -> pd.DataFrame:
        orig_feat, surr_feat = self._dm.efel
        return pd.DataFrame(
            [
                {"feature": feat, "orig": o, "surr": s, "orig-surr": o - s}
                for feat in _MEDIAN_FEATURES
                for o, s in [(_or_nan(np.median, orig_feat.get(feat)), _or_nan(np.median, surr_feat.get(feat)))]
            ]
        ).set_index("feature")


@dataclass
class WaveformMetrics:
    """波形・発火パターン指標（RMSE/MAE・スパイク数・タイミング・ISI）を計算する。"""

    _dm: DynamicMetrics = field(repr=False)

    @cached_property
    def _waveform_error(self) -> pd.DataFrame:
        orig_v, surr_v = self._dm.voltages
        nan = float("nan")
        return pd.DataFrame([
            {"metric": "rmse", "orig": nan, "surr": nan, "orig-surr": float(np.sqrt(np.mean((orig_v - surr_v) ** 2)))},
            {"metric": "mae",  "orig": nan, "surr": nan, "orig-surr": float(np.mean(np.abs(orig_v - surr_v)))},
        ]).set_index("metric")

    @cached_property
    def _spike_counts(self) -> pd.DataFrame:
        orig_peaks, surr_peaks = self._dm.peaks
        o, s = float(len(orig_peaks)), float(len(surr_peaks))
        return pd.DataFrame([
            {"metric": "spike_count", "orig": o, "surr": s, "orig-surr": o - s},
        ]).set_index("metric")

    @cached_property
    def _timing(self) -> pd.DataFrame:
        orig_feat, surr_feat = self._dm.efel
        orig_tfs = orig_feat.get("time_to_first_spike")
        surr_tfs = surr_feat.get("time_to_first_spike")
        nan = float("nan")
        o = float(orig_tfs[0]) if orig_tfs is not None else nan
        s = float(surr_tfs[0]) if surr_tfs is not None else nan
        return pd.DataFrame([
            {"metric": "latency", "orig": o, "surr": s, "orig-surr": o - s if not np.isnan(o + s) else nan},
        ]).set_index("metric")

    @cached_property
    def _isi_stats(self) -> pd.DataFrame:
        orig_feat, surr_feat = self._dm.efel
        orig_isi = orig_feat.get("ISI_values")
        surr_isi = surr_feat.get("ISI_values")
        nan = float("nan")
        o_mean, s_mean = _or_nan(np.mean, orig_isi), _or_nan(np.mean, surr_isi)
        o_std,  s_std  = _or_nan(np.std,  orig_isi), _or_nan(np.std,  surr_isi)
        return pd.DataFrame([
            {"metric": "mean_isi", "orig": o_mean, "surr": s_mean, "orig-surr": o_mean - s_mean if not np.isnan(o_mean + s_mean) else nan},
            {"metric": "std_isi",  "orig": o_std,  "surr": s_std,  "orig-surr": o_std  - s_std  if not np.isnan(o_std  + s_std)  else nan},
        ]).set_index("metric")

    def to_df(self) -> pd.DataFrame:
        return pd.concat([self._waveform_error, self._spike_counts, self._timing, self._isi_stats])

    def compute(self) -> dict:
        df = self.to_df()
        return {
            "rmse":             float(df.loc["rmse", "orig-surr"]),
            "mae":              float(df.loc["mae", "orig-surr"]),
            "orig_spike_count": int(df.loc["spike_count", "orig"]),
            "surr_spike_count": int(df.loc["spike_count", "surr"]),
            "spike_count_diff": abs(int(df.loc["spike_count", "orig-surr"])),
            "latency_error":    abs(float(df.loc["latency", "orig-surr"])),
            "periodicity_gap":  abs(float(df.loc["mean_isi", "orig-surr"])),
            "orig_mean_isi":    float(df.loc["mean_isi", "orig"]),
            "orig_std_isi":     float(df.loc["std_isi",  "orig"]),
            "surr_mean_isi":    float(df.loc["mean_isi", "surr"]),
            "surr_std_isi":     float(df.loc["std_isi",  "surr"]),
        }
