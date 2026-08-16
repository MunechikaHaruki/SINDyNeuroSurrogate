"""波形/スパイク指標の表 (`dynamics.py` の素の計算値 → 列と順に並べる)。

**「何を計算するか」と「どう見せるか」を分ける**: `dynamics.py` は
`DynamicMetrics` を引数に取りスカラー/tuple/dict しか返さない。ここはその値を
どの列名・どの順で並べるかだけを持つ (計算をここで増やさない)。

点軸/run 軸に沿った表 (`sim.artifacts` の run 軸の表) はここに無い — 軸は結果の
関心で、波形ドメインは 1 ペア (原系, 置換系) しか知らない。marimo 非依存。
"""

from __future__ import annotations

import pandas as pd

from .dynamics import (
    DynamicMetrics,
    diff_or_nan,
    spike_feature_values,
    waveform_summary_rows,
)


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
