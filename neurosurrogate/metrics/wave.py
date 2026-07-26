import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import TypeVar

import efel
import numpy as np
import pandas as pd
import xarray as xr

from ..core import access

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
_DIVERGE_V = 1e3  # |V| [mV] の発散判定閾値 (生理的な範囲は ±200 程度)


def diverged(v: np.ndarray) -> bool:
    """電位系列が NaN/inf を含むか、生理的にあり得ない大きさへ振り切れたか。
    サロゲート置換系が数値的に破綻したかの共通基準 (評価ログ・波形図で共用)。"""
    return not bool(np.all(np.isfinite(v))) or float(np.abs(v).max()) > _DIVERGE_V


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


def _pair(fn: Callable[[T], R], pair: tuple[T, T]) -> tuple[R, R]:
    """(orig, surr) ペアに fn を適用して (fn(orig), fn(surr)) を返す。"""
    return fn(pair[0]), fn(pair[1])


def _row(name: str, o: float, s: float, col: str = "metric") -> dict:
    """orig/surr/orig-surr の DataFrame 行 dict を生成。col で index 列名を指定。"""
    return {col: name, "orig": o, "surr": s, "orig-surr": _diff(o, s)}


@dataclass
class DynamicMetrics:
    """電圧・eFEL特徴量を計算するデータ層。下記の純粋関数群から参照される。"""

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

    orig_tmpl, surr_tmpl = (
        _mean_template(v, p) for v, p in zip(dm.voltages, dm.peaks, strict=True)
    )
    return {"spike_shape_corr": _corr_or_nan(orig_tmpl, surr_tmpl)}


def spike_features_df(
    dm: DynamicMetrics,
    spike_orig: int = 0,
    spike_surr: int = 0,
) -> pd.DataFrame:
    """指定 AP の eFEL 特徴量を orig/surr/orig-surr で並べた DataFrame。"""
    orig_feat, surr_feat = dm.efel
    rows = [
        _row(
            feat,
            _at_or_nan(orig_feat.get(feat), spike_orig),
            _at_or_nan(surr_feat.get(feat), spike_surr),
            col="feature",
        )
        for feat in _MEDIAN_FEATURES
    ]
    return pd.DataFrame(rows).set_index("feature")


# ---------------------------------------------------------------------------
# 波形・発火パターン指標（純粋関数群）
# ---------------------------------------------------------------------------

# waveform_summary_df の row 名（原系/置換系の両方が定義される指標）
_ROW_METRICS: list[str] = ["spike_count", "latency", "mean_isi", "std_isi"]
# waveform_summary + spike_shape_corr のキー（両者の比較なので置換系側だけの指標）
_SCALAR_METRICS: list[str] = ["rmse", "mae", "periodicity_gap", "spike_shape_corr"]
# 点軸メトリクス図で選べる metric の**単一源**。UI の選択肢も `extract_metric` の
# 受理集合もここから引く (別々に並べると、生成されないキーを選べてしまい黙って
# nan 図が出る)。
METRIC_KEYS: list[str] = _ROW_METRICS + _SCALAR_METRICS


def _waveform_error(dm: DynamicMetrics) -> dict:
    """RMSE/MAE の波形誤差スカラー。"""
    orig_v, surr_v = dm.voltages
    return {
        "rmse": float(np.sqrt(np.mean((orig_v - surr_v) ** 2))),
        "mae": float(np.mean(np.abs(orig_v - surr_v))),
    }


def _latency(dm: DynamicMetrics) -> tuple[float, float]:
    return _pair(lambda f: _at_or_nan(f.get("time_to_first_spike"), 0), dm.efel)


def _isi_stat(dm: DynamicMetrics, fn) -> tuple[float, float]:
    isi = _pair(lambda f: f.get("ISI_values"), dm.efel)
    return _pair(lambda a: _or_nan(fn, a), isi)


def waveform_summary_df(dm: DynamicMetrics) -> pd.DataFrame:
    """spike_count / latency / mean_isi / std_isi を縦に並べた DataFrame。"""
    o_n, s_n = n_spikes(dm)
    return pd.DataFrame(
        [
            _row("spike_count", float(o_n), float(s_n)),
            _row("latency", *_latency(dm)),
            _row("mean_isi", *_isi_stat(dm, np.mean)),
            _row("std_isi", *_isi_stat(dm, np.std)),
        ]
    ).set_index("metric")


def waveform_summary(dm: DynamicMetrics) -> dict:
    """波形誤差 (rmse/mae) + 発火周期のズレ (periodicity_gap)。"""
    return {
        **_waveform_error(dm),
        "periodicity_gap": abs(_diff(*_isi_stat(dm, np.mean))),
    }


# ---------------------------------------------------------------------------
# 統合レポート（DataFrame 組立まで metrics 側で完結。呼び出し側は表示/保存だけ）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaveReport:
    """波形+スパイク指標を統合した評価レポート。df をそのまま表示/保存へ流す。"""

    df_metrics: pd.DataFrame  # 波形行 (+ 指定 spike が両信号にあればその特徴量)
    df_scalar: pd.DataFrame  # 全スカラーを縦持ち


def wave_report(
    dm: DynamicMetrics,
    spike_orig: int = 0,
    spike_surr: int = 0,
) -> WaveReport:
    """dm から波形/スパイク指標を計算し DataFrame まで組み立てて返す。指定した
    spike index が両信号の範囲内にあるときだけ、その AP の特徴量と形状相関を足す。"""
    n_orig, n_surr = n_spikes(dm)
    df_metrics = waveform_summary_df(dm)
    scalar = waveform_summary(dm)
    if 0 <= spike_orig < n_orig and 0 <= spike_surr < n_surr:
        df_spike = spike_features_df(dm, spike_orig=spike_orig, spike_surr=spike_surr)
        df_spike.index.name = "metric"
        df_metrics = pd.concat([df_metrics, df_spike])
        scalar.update(spike_shape_corr(dm))
    return WaveReport(
        df_metrics=df_metrics,
        df_scalar=pd.DataFrame(scalar.items(), columns=["metric", "value"]).set_index(
            "metric"
        ),
    )


def extract_metric(dm: DynamicMetrics, metric_key: str) -> tuple[float | None, float]:
    """指定 metric の (orig, surr)。両者の比較で決まるスカラー metric に原系の値は
    無い → orig は None。未知キーは KeyError (選択肢は `METRIC_KEYS` が単一源で、
    そこに載っていて取り出せないキーがあれば黙って nan を返さず落とす)。"""
    if metric_key in _ROW_METRICS:
        df = waveform_summary_df(dm)
        return (
            float(df.loc[metric_key, "orig"]),  # type: ignore[arg-type]
            float(df.loc[metric_key, "surr"]),  # type: ignore[arg-type]
        )
    return None, float({**waveform_summary(dm), **spike_shape_corr(dm)}[metric_key])
