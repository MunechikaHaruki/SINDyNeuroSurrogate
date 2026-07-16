from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..core import access
from ..currents import CURRENT_MAP
from ..models import MCMODELS
from .engine import _JP_FONT, error_fig

if TYPE_CHECKING:
    from ..metrics.eval_sweep import CurrentSweepConfig, SweepEval


def current_preview_fig(current_type: str, dt: float, params: dict) -> Figure:
    """電流波形プレビュー。構築失敗は error_fig。marimo 非依存。"""
    try:
        i_ext = CURRENT_MAP[current_type](**params)(dt)
    except Exception as e:  # noqa: BLE001
        return error_fig(f"build失敗: {e}")
    t = np.arange(len(i_ext)) * dt
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(t, i_ext, lw=0.8)
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("I_ext [μA/cm²]")
    ax.set_title(f"{current_type} preview")
    fig.tight_layout()
    return fig


def sweep_fig(
    data: pd.DataFrame,
    cfg: CurrentSweepConfig,
    comp_name: str,
    metric_key: str,
    run_labels: dict[str, str],
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """sweep メトリクス折れ線 (Original + surrogate 各 run)。marimo 非依存。
    run_labels は rid→表示名 (順序=描画順、keys=run_ids)。"""
    fig, ax = plt.subplots()
    if "original" in data.columns:
        ax.plot(data["amplitude"], data["original"], "k-o", label="Original", zorder=3)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (rid, label) in enumerate(run_labels.items()):
        ax.plot(
            data["amplitude"],
            data[rid],
            marker="s",
            linestyle="--",
            color=colors[idx % len(colors)],
            label=label,
        )

    ax.set_xlabel(cfg.sweep_param)
    ax.set_ylabel(metric_key)
    ax.set_title(f"{cfg.sweep_param} sweep — {metric_key} ({comp_name})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    return fig


def sweep_trace_grid_fig(
    sweep: SweepEval,
    comp_name: str,
    run_labels: dict[str, str],
) -> Figure:
    """列=掃引 amp の波形格子。行1=I_ext、行2以降=各 run の V 波形 (orig 重畳)。
    run_labels は rid→表示名 (順序=行順、keys=run_ids)。marimo 非依存。"""
    comp_id = MCMODELS[sweep.model_name].name_to_idx(comp_name)
    n_col = len(sweep.amp_datasets)
    n_row = 1 + len(run_labels)
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(2.6 * n_col, 1.5 * n_row),
        squeeze=False,
        sharex=True,
    )
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
        for r, rid in enumerate(run_labels, start=1):
            ax = axes[r][c]
            ax.set_ylim(*v_ylim)
            ax.plot(
                access.time(orig_ds),
                access.potential(orig_ds, comp_id),
                "k-",
                lw=0.7,
                label="Original",
            )
            surr_v = access.potential(surr_datasets[rid], comp_id)
            if not np.all(np.isfinite(surr_v)) or float(np.abs(surr_v).max()) > 1e4:
                ax.text(
                    0.5,
                    0.5,
                    "発散",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="red",
                    fontproperties=_JP_FONT,
                )
                continue
            ax.plot(
                access.time(surr_datasets[rid]),
                surr_v,
                "--",
                lw=0.7,
                color="tab:red",
                label="surrogate",
            )
    axes[0][0].set_ylabel("I_ext")
    for r, label in enumerate(run_labels.values(), start=1):
        axes[r][0].set_ylabel(label, fontproperties=_JP_FONT)
    for c in range(n_col):
        axes[-1][c].set_xlabel("t [ms]")
    axes[1][0].legend(loc="upper right", fontsize="x-small")
    fig.suptitle(f"amp sweep waveform ({comp_name})", fontproperties=_JP_FONT)
    fig.tight_layout()
    return fig
