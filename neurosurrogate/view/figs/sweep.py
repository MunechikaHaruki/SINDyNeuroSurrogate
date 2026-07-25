"""掃引結果 (`SweepEval`) の図: メトリクス折れ線と amp×run の波形格子。

軸名も run 軸ラベルも結果 (`SweepEval.spec` / `run_labels`) から引く = 呼び出し側で
作り直さない。marimo 非依存。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ...core import access
from ...metrics.wave import diverged
from ..engine import new_figure, place_legend

if TYPE_CHECKING:
    from ...metrics.eval import SweepEval


def sweep_fig(
    sweep: SweepEval,
    comp_name: str,
    metric_key: str,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """sweep メトリクス折れ線 (Original + surrogate 各 run)。marimo 非依存。
    run 軸ラベルも掃引軸名も結果 (sweep) から引く = 別引数で持ち回らない。"""
    data = sweep.metrics_df(comp_name, metric_key)
    labels = sweep.run_labels
    spec = sweep.spec
    fig = new_figure()
    ax = fig.subplots()
    if "original" in data.columns:
        ax.plot(data["amplitude"], data["original"], "k-o", label="Original", zorder=3)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, label in enumerate(labels):
        ax.plot(
            data["amplitude"],
            data[label],
            marker="s",
            linestyle="--",
            color=colors[idx % len(colors)],
            label=label,
        )

    ax.set_xlabel(spec.sweep_param)
    ax.set_ylabel(metric_key)
    ax.set_title(f"{spec.sweep_param} sweep — {metric_key} ({comp_name})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    place_legend(ax)
    return fig


def sweep_trace_grid_fig(sweep: SweepEval, comp_name: str) -> Figure:
    """列=掃引 amp の波形格子。行1=I_ext、行2以降=各 run の V 波形 (orig 重畳)。
    行順は結果の run 軸 (`sweep.run_labels`) そのもの。marimo 非依存。"""
    comp_id = sweep.spec.net.name_to_idx(comp_name)
    labels = sweep.run_labels
    n_col = len(sweep.amp_datasets)
    n_row = 1 + len(labels)
    fig = new_figure(figsize=(2.6 * n_col, 1.5 * n_row))
    axes = fig.subplots(n_row, n_col, squeeze=False, sharex=True)
    # y レンジは発散しない Original 電位の全 amp min/max から決める (ニューロン挙動を
    # 捉えるレンジ)。全 V 行で共有し、発散 surrogate はこのレンジで頭打ちにする。
    orig_vs = [
        access.potential(orig_ds, comp_id) for _, orig_ds, _ in sweep.amp_datasets
    ]
    lo = min(float(v.min()) for v in orig_vs)
    hi = max(float(v.max()) for v in orig_vs)
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    v_ylim = (lo - pad, hi + pad)

    for c, (amp, orig_ds, surr_datasets) in enumerate(sweep.amp_datasets):
        axes[0][c].plot(*access.i_ext(orig_ds), lw=0.8, color="tab:gray")
        axes[0][c].set_title(f"amp={amp:.3g}")
        for r, label in enumerate(labels, start=1):
            ax = axes[r][c]
            ax.set_ylim(*v_ylim)
            ax.plot(
                access.time(orig_ds),
                access.potential(orig_ds, comp_id),
                "k-",
                lw=0.7,
                label="Original",
            )
            surr_v = access.potential(surr_datasets[label], comp_id)
            if diverged(surr_v):
                ax.text(
                    0.5,
                    0.5,
                    "diverged",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="red",
                )
                continue
            ax.plot(
                access.time(surr_datasets[label]),
                surr_v,
                "--",
                lw=0.7,
                color="tab:red",
                label="surrogate",
            )
    axes[0][0].set_ylabel("I_ext")
    for r, label in enumerate(labels, start=1):
        axes[r][0].set_ylabel(label)
    for c in range(n_col):
        axes[-1][c].set_xlabel("t [ms]")
    place_legend(axes[1][-1])
    fig.suptitle(f"amp sweep waveform ({comp_name})")
    return fig
