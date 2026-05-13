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
        "orig_spike_count": len(orig_peaks),          # 元モデルのスパイク総数
        "surr_spike_count": len(surr_peaks),           # サロゲートのスパイク総数
        "spike_count_diff": abs(len(orig_peaks) - len(surr_peaks)),  # スパイク数の絶対差
        "latency_error": latency_error,                # 第1スパイクまでのレイテンシ差 [ms]
        **_isi_stats(orig_isi, "orig"),                # 元モデルの ISI 平均・標準偏差
        **_isi_stats(surr_isi, "surr"),                # サロゲートの ISI 平均・標準偏差
        "periodicity_gap": periodicity_gap,            # 平均 ISI の差（発火周期のずれ）[ms]
        # ピーク・閾値
        "median_amp_error": _median_diff("peak_voltage"),       # ピーク電圧の中央値差 [mV]
        "median_amplitude_error": _median_diff("AP_amplitude"),  # 閾値〜ピーク振幅の中央値差 [mV]
        "median_threshold_error": _median_diff("AP_begin_voltage"),  # 発火閾値電圧の中央値差 [mV]
        # dV/dt（活動電位の鋭さ）
        "median_max_dvdt_error": _median_diff("AP_rise_rate"),  # 最大上昇速度の中央値差 [mV/ms]
        "median_min_dvdt_error": _median_diff("AP_fall_rate"),  # 最大下降速度の中央値差 [mV/ms]
        # スパイク内タイミング
        "median_half_width_error": _median_diff("AP_duration_half_width"),  # 半値幅の中央値差 [ms]
        "median_rise_time_error": _median_diff("AP_rise_time"),   # 上昇時間（閾値→ピーク）の中央値差 [ms]
        "median_fall_time_error": _median_diff("AP_fall_time"),   # 下降時間（ピーク→閾値）の中央値差 [ms]
        # 後過分極（AHP）
        "median_ahp_depth_error": _median_diff("AHP_depth"),         # AHP 深さの中央値差 [mV]
        "median_ahp_time_error": _median_diff("AHP_time_from_peak"),  # ピーク〜AHP 最深部までの時間差 [ms]
    }


def _calc_waveform_metrics(orig_v, surr_v):
    return {
        "rmse": float(np.sqrt(np.mean((orig_v - surr_v) ** 2))),  # 全時系列の二乗平均誤差 [mV]
        "mae": float(np.mean(np.abs(orig_v - surr_v))),           # 全時系列の平均絶対誤差 [mV]
    }


def _calc_spike_shape_corr(
    orig_v: np.ndarray,
    surr_v: np.ndarray,
    orig_peaks: list[int],
    surr_peaks: list[int],
    half_win: int = 50,
) -> dict:
    """元モデルとサロゲートの平均スパイクテンプレート間の Pearson 相関（1に近いほど形状が一致）。"""

    def _mean_template(v, peaks):
        snippets = []
        for p in peaks:
            lo, hi = p - half_win, p + half_win + 1
            if lo >= 0 and hi <= len(v):
                snippets.append(v[lo:hi])
        if not snippets:
            return None
        return np.mean(snippets, axis=0)

    orig_tmpl = _mean_template(orig_v, orig_peaks)
    surr_tmpl = _mean_template(surr_v, surr_peaks)

    if orig_tmpl is None or surr_tmpl is None:
        return {"spike_shape_corr": float("nan")}

    corr = float(np.corrcoef(orig_tmpl, surr_tmpl)[0, 1])
    return {"spike_shape_corr": corr}  # スパイク波形テンプレートの相関係数（−1〜1）


_EFEL_FEATURES = [
    "peak_indices",
    "ISI_values",
    "time_to_first_spike",
    "peak_voltage",
    "AP_rise_rate",
    "AP_fall_rate",
    # Single spike morphology
    "AP_amplitude",
    "AP_begin_voltage",
    "AP_duration_half_width",
    "AP_rise_time",
    "AP_fall_time",
    "AHP_depth",
    "AHP_time_from_peak",
]


def _to_efel_trace(voltage: np.ndarray, dt: float) -> dict:
    time = np.arange(len(voltage), dtype=float) * dt
    # stim_start を 1 サンプル後ろにずらして AHP baseline 計算に必要な区間を確保
    stim_start = time[min(1, len(time) - 1)]
    return {
        "T": time,
        "V": voltage.astype(float),
        "stim_start": [stim_start],
        "stim_end": [time[-1]],
    }


def calc_dynamic_metrics(original, surrogate, comp_id, dt):
    orig_v = original["vars"].sel(gate=False, comp_id=comp_id).to_numpy().squeeze()
    surr_v = surrogate["vars"].sel(gate=False, comp_id=comp_id).to_numpy().squeeze()

    orig_feat, surr_feat = efel.get_feature_values(
        [_to_efel_trace(orig_v, dt), _to_efel_trace(surr_v, dt)],
        _EFEL_FEATURES,
    )

    # numpy 配列に対して or を使うと曖昧なため is None で判定
    _peaks = orig_feat.get("peak_indices")
    orig_peaks = list(_peaks) if _peaks is not None else []
    _peaks = surr_feat.get("peak_indices")
    surr_peaks = list(_peaks) if _peaks is not None else []

    return {
        **_calc_waveform_metrics(orig_v, surr_v),
        **_calc_spike_metrics(orig_feat, surr_feat),
        **_calc_spike_shape_corr(orig_v, surr_v, orig_peaks, surr_peaks),
    }
