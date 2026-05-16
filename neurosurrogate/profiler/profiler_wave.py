import warnings
from dataclasses import dataclass, field
from typing import Any

import efel
import numpy as np

_MEDIAN_ERROR_FEATURES: list[str] = [
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
    *_MEDIAN_ERROR_FEATURES,
]


def _to_list(val) -> list:
    return [] if val is None else list(val)


def _or_nan(fn, arr) -> float:
    if arr is None or len(arr) == 0:
        return float("nan")
    return float(fn(arr))


@dataclass
class DynamicMetrics:
    original: Any = field(repr=False)
    surrogate: Any = field(repr=False)
    comp_id: int
    dt: float

    def _voltages(self) -> tuple[np.ndarray, np.ndarray]:
        orig_v = self.original["vars"].sel(gate=False, comp_id=self.comp_id).to_numpy().squeeze()
        surr_v = self.surrogate["vars"].sel(gate=False, comp_id=self.comp_id).to_numpy().squeeze()
        return orig_v, surr_v

    def _efel_features(self, orig_v: np.ndarray, surr_v: np.ndarray) -> tuple[dict, dict]:
        def _to_trace(v: np.ndarray) -> dict:
            time = np.arange(len(v), dtype=float) * self.dt
            # stim_start を 1 サンプル後ろにずらして AHP baseline 計算に必要な区間を確保
            return {"T": time, "V": v.astype(float), "stim_start": [time[min(1, len(time) - 1)]], "stim_end": [time[-1]]}

        with warnings.catch_warnings():
            # スパイクなし時に eFEL が RuntimeWarning を出すが、nan に変換するため問題ない
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"efel\.*")
            orig_feat, surr_feat = efel.get_feature_values(  # type: ignore[reportCallIssue]
                [_to_trace(orig_v), _to_trace(surr_v)],
                _EFEL_FEATURES,
            )
        return orig_feat, surr_feat

    def waveform_metrics(self) -> dict:
        """波形全体指標（rmse/mae + 発火パターン）を返す。"""

        def _isi_stats(isi, prefix: str) -> dict:
            return {
                f"{prefix}_mean_isi": _or_nan(np.mean, isi),
                f"{prefix}_std_isi": _or_nan(np.std, isi),
            }

        orig_v, surr_v = self._voltages()
        orig_feat, surr_feat = self._efel_features(orig_v, surr_v)
        orig_peaks = _to_list(orig_feat.get("peak_indices"))
        surr_peaks = _to_list(surr_feat.get("peak_indices"))

        orig_isi = orig_feat.get("ISI_values")
        surr_isi = surr_feat.get("ISI_values")
        orig_tfs = orig_feat.get("time_to_first_spike")
        surr_tfs = surr_feat.get("time_to_first_spike")
        latency_error = (
            float(abs(orig_tfs[0] - surr_tfs[0]))
            if orig_tfs is not None and surr_tfs is not None
            else float("nan")
        )
        periodicity_gap = (
            float(abs(np.mean(orig_isi) - np.mean(surr_isi)))
            if orig_isi is not None and surr_isi is not None
            else float("nan")
        )
        return {
            "rmse": float(np.sqrt(np.mean((orig_v - surr_v) ** 2))),
            "mae": float(np.mean(np.abs(orig_v - surr_v))),
            "orig_spike_count": len(orig_peaks),
            "surr_spike_count": len(surr_peaks),
            "spike_count_diff": abs(len(orig_peaks) - len(surr_peaks)),
            "latency_error": latency_error,
            **_isi_stats(orig_isi, "orig"),
            **_isi_stats(surr_isi, "surr"),
            "periodicity_gap": periodicity_gap,
        }

    def spike_shape_metrics(self) -> dict:
        """スパイク形状指標（median AP誤差、spike_shape_corr）を返す。"""

        def _median_or_nan(arr) -> float:
            return _or_nan(np.median, arr)

        def _spike_shape_corr(orig_v, surr_v, orig_peaks, surr_peaks, half_win: int = 50) -> dict:
            """平均スパイクテンプレート間の Pearson 相関（1に近いほど形状が一致）。"""
            def _mean_template(v, peaks):
                snippets = [v[p - half_win: p + half_win + 1] for p in peaks if p - half_win >= 0 and p + half_win + 1 <= len(v)]
                return np.mean(snippets, axis=0) if snippets else None

            orig_tmpl = _mean_template(orig_v, orig_peaks)
            surr_tmpl = _mean_template(surr_v, surr_peaks)
            if orig_tmpl is None or surr_tmpl is None:
                return {"spike_shape_corr": float("nan")}
            return {"spike_shape_corr": float(np.corrcoef(orig_tmpl, surr_tmpl)[0, 1])}

        orig_v, surr_v = self._voltages()
        orig_feat, surr_feat = self._efel_features(orig_v, surr_v)
        orig_peaks = _to_list(orig_feat.get("peak_indices"))
        surr_peaks = _to_list(surr_feat.get("peak_indices"))

        def _md(key: str) -> float:
            return abs(_median_or_nan(orig_feat.get(key)) - _median_or_nan(surr_feat.get(key)))

        return {
            **{f"median_{feat}_error": _md(feat) for feat in _MEDIAN_ERROR_FEATURES},
            **_spike_shape_corr(orig_v, surr_v, orig_peaks, surr_peaks),
        }
