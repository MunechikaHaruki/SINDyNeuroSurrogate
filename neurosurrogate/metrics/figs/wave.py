"""波形/スパイク指標の DataFrame 化 (`metrics/wave.py` の素の計算値 → 表として並べる)。

**「何を計算するか」と「どう見せるか」を分ける**: `metrics/wave.py` は
`DynamicMetrics` を引数に取り、スカラー/tuple/dict しか返さない (marimo/mlflow
非依存の純粋計算層)。ここは描画層 = その値をどの列名・どの順で DataFrame に
並べるかだけを持つ (計算をここで増やさない)。marimo 非依存。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from ..wave import (
    DynamicMetrics,
    diff_or_nan,
    dm_at,
    extract_metric,
    n_spikes,
    spike_feature_values,
    spike_shape_corr,
    waveform_summary,
    waveform_summary_rows,
)

if TYPE_CHECKING:
    from ...eval.eval import EvalGrid


def _row(name: str, o: float, s: float, col: str = "metric") -> dict:
    """orig/surr/orig-surr の DataFrame 行 dict を生成。col で index 列名を指定。"""
    return {col: name, "orig": o, "surr": s, "orig-surr": diff_or_nan(o, s)}


def waveform_summary_df(dm: DynamicMetrics) -> pd.DataFrame:
    """spike_count / latency / mean_isi / std_isi を縦に並べた DataFrame。"""
    rows = waveform_summary_rows(dm)
    return pd.DataFrame([_row(name, o, s) for name, (o, s) in rows.items()]).set_index(
        "metric"
    )


def spike_features_df(
    dm: DynamicMetrics,
    spike_orig: int = 0,
    spike_surr: int = 0,
) -> pd.DataFrame:
    """指定 AP の eFEL 特徴量を orig/surr/orig-surr で並べた DataFrame。"""
    values = spike_feature_values(dm, spike_orig=spike_orig, spike_surr=spike_surr)
    rows = [_row(feat, o, s, col="feature") for feat, (o, s) in values.items()]
    return pd.DataFrame(rows).set_index("feature")


def metrics_df(grid: EvalGrid, comp_name: str, metric_key: str) -> pd.DataFrame:
    """点軸に沿った metric の DataFrame (列=run 軸)。原系の値は run に依らない
    ので `original` 列 1 本へ畳む。`EvalGrid` 自身のメソッドにしない (計算結果の
    純粋データ型に評価/DataFrame 化の関心を持たせない)。"""
    comp_id = grid.spec.net.name_to_idx(comp_name)
    rows: list[dict] = []
    for i, point in enumerate(grid.points):
        row: dict = {"point": point.value}
        for run_label in point.surrogates:
            orig, surr = extract_metric(dm_at(grid, i, run_label, comp_id), metric_key)
            row[run_label] = surr
            if orig is not None:
                row["original"] = orig  # run に依らない = 同じ値の上書き
        rows.append(row)
    return pd.DataFrame(rows)


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
