"""**波形ドメイン**: 原系と置換系の 1 ペアを突き合わせて「どれだけ合っているか」を
指標と図に落とす (研究の主張のうち波形再現性の側)。

- `dynamics.py` — eFEL スパイク特徴量・波形誤差の計算 (素の値だけ返す)
- `_tables.py` — その値を表に並べる
- `_figures.py` — 波形/差分/相平面の図

ここは**常に 1 ペア** (原系, 置換系) しか知らない: 点軸 (掃引) も run 軸 (どの
surrogate) も持たない = 軸を掛けるのは結果の関心 (`neurosurrogate.report`)。
marimo/mlflow 非依存。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
import xarray as xr
from matplotlib import rcParams

from ..plotting import Artifact, collect, draw_engine
from ._figures import attractor_fig, panels_diff, panels_simple
from ._figures import current_preview_fig as current_preview_fig
from ._tables import spike_features_df, waveform_summary_df
from .dynamics import DynamicMetrics, n_spikes, spike_shape_corr, waveform_summary
from .dynamics import dm_of as dm_of  # 軸を掛ける側 (sim.report.report) が使う
from .dynamics import extract_metric as extract_metric


def cell_figs(
    original: xr.Dataset,
    surrogate: xr.Dataset,
    comp_id: int,
    latent: Callable[[], xr.Dataset],
    comps: Sequence[int] | None = None,
    i_ext_ylim: tuple[float, float] | None = None,
) -> list[Artifact]:
    """1 ペアの全図を識別子付きで一括生成 (失敗の畳み込みは `collect`)。
    呼び出し側は種別を知らず `Artifact` 列を保存/表示に流すだけ。

    comp_id=比較対象 (diff/attractor は 1 comp の話)、comps=全 comp を並べる図
    (simple) の表示制限。i_ext_ylim=train_raw.png と軸を揃えたいとき渡す (発表用)。

    `latent` (原系ゲートの潜在射影) は **callable で受けて lazy 参照**: 学習ドメイン
    外 comp では raise するので diff/attractor でのみ評価する (simple は呼ばない)。
    """
    return collect(
        {
            "diff": lambda: draw_engine(
                panels_diff(original, latent(), surrogate, comp_id, i_ext_ylim),
                figsize=(
                    rcParams["figure.figsize"][0],
                    rcParams["figure.figsize"][1] * 1.3,
                ),
            ),
            "simple": lambda: draw_engine(panels_simple(original, comps)),
            "attractor": lambda: attractor_fig(latent(), surrogate, comp_id),
        }
    )


def wave_report(
    dm: DynamicMetrics,
    spike_orig: int = 0,
    spike_surr: int = 0,
) -> list[Artifact]:
    """dm から波形/スパイク指標を計算し表まで組み立てて返す (metrics=波形行 + 指定
    spike があればその特徴量、metrics_scalar=全スカラーを縦持ち)。指定した spike
    index が両信号の範囲内にあるときだけ、その AP の特徴量と形状相関を足す。"""
    n_orig, n_surr = n_spikes(dm)
    df_metrics = waveform_summary_df(dm)
    scalar = waveform_summary(dm)
    if 0 <= spike_orig < n_orig and 0 <= spike_surr < n_surr:
        df_spike = spike_features_df(dm, spike_orig=spike_orig, spike_surr=spike_surr)
        df_spike.index.name = "metric"
        df_metrics = pd.concat([df_metrics, df_spike])
        scalar.update(spike_shape_corr(dm))
    df_scalar = pd.DataFrame(scalar.items(), columns=["metric", "value"]).set_index(
        "metric"
    )
    return [Artifact("metrics", df_metrics), Artifact("metrics_scalar", df_scalar)]
