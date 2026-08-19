"""**1 ペアの詳細成果物**: 入力電流プレビューと、原系/置換系の比較。

どのペアを描くかは呼び出し側が選び、ここは Dataset だけを受ける (結果型
`SeriesResults` を知らない)。**1 ペアから何を出すかはここが持つ** =
`detail_artifacts` が集合ごと返し、合流点はそれを受け取って段へ書くだけ。
marimo 非依存。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import rcParams

from ...artifact.model import Artifact
from ...artifact.plotting import (
    PanelSpec,
    TraceSpec,
    draw_engine,
    new_figure,
    place_legend,
)
from ...core import access
from ...core.access import POTENTIAL_VAR
from ..waveform import DynamicMetrics, n_spikes, spike_shape_corr, waveform_summary
from ._tables import spike_features_df, waveform_summary_df


def diff_artifact(
    original: xr.Dataset,
    preprocessed: xr.Dataset,
    surrogate: xr.Dataset,
    comp_id: int,
) -> Artifact:
    """1 ペアの差分図を単一成果物として返す。"""
    return Artifact(
        "diff",
        draw_engine(
            _panels_diff(original, preprocessed, surrogate, comp_id),
            figsize=(
                rcParams["figure.figsize"][0],
                rcParams["figure.figsize"][1] * 1.3,
            ),
        ),
    )


def simple_artifact(
    original: xr.Dataset, comps: Sequence[int] | None = None
) -> Artifact:
    return Artifact("simple", draw_engine(_panels_simple(original, comps)))


def _has_spike_pair(dm: DynamicMetrics, spike_orig: int, spike_surr: int) -> bool:
    """指定されたスパイク対が両系列に存在するか。

    **スパイクが 1 本も無い**のは正常 (静止した置換系など) なので、その場合は特徴量を
    落として指標だけ返す。一方、スパイクはあるのに範囲外の番号を指されたのは指定ミス
    なので、黙って行を落とさず知らせる (旧実装はどちらも同じく無言で落としていた)。
    """
    n_orig, n_surr = n_spikes(dm)
    if not n_orig or not n_surr:
        return False
    if not (0 <= spike_orig < n_orig and 0 <= spike_surr < n_surr):
        raise ValueError(
            f"spike index が範囲外 "
            f"(orig {spike_orig}/{n_orig}, surr {spike_surr}/{n_surr})"
        )
    return True


def metrics_artifact(dm: DynamicMetrics, spike_orig: int, spike_surr: int) -> Artifact:
    """波形指標と、指定されたスパイクの特徴量を表にする。"""
    metrics = waveform_summary_df(dm)
    if _has_spike_pair(dm, spike_orig, spike_surr):
        spike = spike_features_df(dm, spike_orig=spike_orig, spike_surr=spike_surr)
        spike.index.name = "metric"
        metrics = pd.concat([metrics, spike])
    return Artifact("metrics", metrics)


def metrics_scalar_artifact(
    dm: DynamicMetrics, spike_orig: int, spike_surr: int
) -> Artifact:
    metrics = waveform_summary(dm)
    if _has_spike_pair(dm, spike_orig, spike_surr):
        metrics.update(spike_shape_corr(dm))
    return Artifact(
        "metrics_scalar",
        pd.DataFrame(metrics.items(), columns=["metric", "value"]).set_index("metric"),
    )


def _panels_simple(
    ds: xr.Dataset, comps: Sequence[int] | None = None
) -> list[PanelSpec]:
    """全 comp の波形。comps を渡すとその comp だけに絞る (None=全部)。
    traub19 のような多 comp モデルは全部重ねると読めないため。"""
    comp_ids = [int(i) for i in access.comp_ids(ds) if comps is None or int(i) in comps]
    multi = len(comp_ids) > 1
    spec: list[PanelSpec] = [
        PanelSpec("I_ext", [TraceSpec(*access.i_ext(ds))]),
    ]

    if access.has_i_internal(ds):
        spec.append(
            PanelSpec(
                "I_internal",
                [
                    TraceSpec(*access.i_internal(ds, i), label=f"Comp {i}")
                    for i in comp_ids
                ],
            )
        )

    spec.append(
        PanelSpec(
            "v(t) [mV]",
            [
                TraceSpec(
                    *access.trace(ds, i, POTENTIAL_VAR),
                    label=f"v (Comp {i})" if multi else None,
                )
                for i in comp_ids
            ],
        )
    )

    gate_traces = [
        TraceSpec(*access.trace(ds, i, v), label=f"{v} (Comp {i})")
        for i in comp_ids
        for v in access.gate_variables(ds, i)
    ]
    if gate_traces:
        spec.append(PanelSpec("Gates / Latent", gate_traces))

    return spec


def _panels_diff(
    original: xr.Dataset,
    preprocessed: xr.Dataset,
    surrogate: xr.Dataset,
    comp_id: int,
) -> list[PanelSpec]:
    return [
        PanelSpec(
            "I_ext(t)\n[μA/cm²]",
            [TraceSpec(*access.i_ext(original), color="gold")],
            # 評価刺激が読める発表用レンジ (train_raw と揃えると学習パルスの最大値に
            # 引っ張られ、評価の step が潰れる)。
            ylim=(0.0, 5.0),
        ),
        PanelSpec(
            "v(t) [mV]",
            [
                TraceSpec(
                    *access.trace(original, comp_id, POTENTIAL_VAR),
                    label="orig v",
                    color="blue",
                ),
                TraceSpec(
                    *access.trace(surrogate, comp_id, POTENTIAL_VAR),
                    label="surr v",
                    color="red",
                    style="--",
                ),
            ],
        ),
        *[
            PanelSpec(
                latent,
                [
                    TraceSpec(
                        *access.trace(preprocessed, comp_id, latent),
                        label=f"orig {latent}",
                        color="blue",
                    ),
                    TraceSpec(
                        *access.trace(surrogate, comp_id, latent),
                        label=f"surr {latent}",
                        color="red",
                        style="--",
                    ),
                ],
            )
            for latent in access.latent_variables(preprocessed)
        ],
    ]


def attractor_artifact(
    orig_ds: xr.Dataset, surr_ds: xr.Dataset, comp_id: int
) -> Artifact:
    """相平面 (V × 第1潜在) の重ね描き。原系と置換系のダイナミクス一致度を見る。
    変数が無い comp では KeyError が呼び出し元へ伝播する。"""
    x_var, y_var = access.POTENTIAL_VAR, access.latent_vars(1)[0]

    def trajectory(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        # access.trace は (t, y) を返す。相平面は値のみ使う
        return tuple(access.trace(ds, comp_id, v)[1] for v in (x_var, y_var))  # type: ignore[return-value]

    fig = new_figure()
    ax = fig.subplots()
    o_x, o_y = trajectory(orig_ds)
    s_x, s_y = trajectory(surr_ds)
    # 原系は黒で「正解」の形。alpha を下げて重なりを見やすくする
    ax.plot(
        o_x, o_y, color="black", linewidth=1.2, alpha=0.6, label="Original (Target)"
    )
    ax.plot(
        s_x, s_y, color="crimson", linewidth=1.0, alpha=0.8, label="Surrogate (SINDy)"
    )
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(f"Attractor Comparison (Comp {comp_id})")
    # ランダム電流だと軌道がボヤける → グリッドがあると位置関係を追いやすい
    ax.grid(True, linestyle=":", alpha=0.5)
    place_legend(ax)
    return Artifact("attractor", fig)
