import efel
import numpy as np


def _calc_spike_metrics(orig_feat: dict, surr_feat: dict) -> dict:
    def _median_or_nan(arr) -> float:
        if arr is None or len(arr) == 0:
            return float("nan")
        return float(np.median(arr))

    def _median_diff(key) -> float:
        return abs(
            _median_or_nan(orig_feat.get(key)) - _median_or_nan(surr_feat.get(key))
        )

    def _safe_float(val) -> float:
        if val is None:
            return float("nan")
        return float(val)

    def _isi_stats(isi, prefix: str) -> dict:
        return {
            f"{prefix}_mean_isi": _safe_float(
                np.mean(isi) if isi is not None else None
            ),
            f"{prefix}_std_isi": _safe_float(np.std(isi) if isi is not None else None),
        }

    def _to_list(val):
        if val is None:
            return []
        return list(val)

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
        "orig_spike_count": len(orig_peaks),
        "surr_spike_count": len(surr_peaks),
        "spike_count_diff": abs(len(orig_peaks) - len(surr_peaks)),
        "latency_error": latency_error,
        **_isi_stats(orig_isi, "orig"),
        **_isi_stats(surr_isi, "surr"),
        "periodicity_gap": periodicity_gap,
        "median_amp_error": _median_diff("peak_voltage"),
        "median_max_dvdt_error": _median_diff("AP_rise_rate"),
        "median_min_dvdt_error": _median_diff("AP_fall_rate"),
    }


def _calc_waveform_metrics(orig_v, surr_v):
    return {
        "rmse": float(np.sqrt(np.mean((orig_v - surr_v) ** 2))),
        "mae": float(np.mean(np.abs(orig_v - surr_v))),
    }


_EFEL_FEATURES = [
    "peak_indices",
    "ISI_values",
    "time_to_first_spike",
    "peak_voltage",
    "AP_rise_rate",
    "AP_fall_rate",
]


def _to_efel_trace(voltage: np.ndarray, dt: float) -> dict:
    time = np.arange(len(voltage), dtype=float) * dt
    return {
        "T": time,
        "V": voltage.astype(float),
        "stim_start": [time[0]],
        "stim_end": [time[-1]],
    }


def calc_dynamic_metrics(original, surrogate, comp_id, dt):
    orig_v = original["vars"].sel(gate=False, comp_id=comp_id).to_numpy().squeeze()
    surr_v = surrogate["vars"].sel(gate=False, comp_id=comp_id).to_numpy().squeeze()

    orig_feat, surr_feat = efel.get_feature_values(
        [_to_efel_trace(orig_v, dt), _to_efel_trace(surr_v, dt)],
        _EFEL_FEATURES,
    )
    return {
        **_calc_waveform_metrics(orig_v, surr_v),
        **_calc_spike_metrics(orig_feat, surr_feat),
    }
