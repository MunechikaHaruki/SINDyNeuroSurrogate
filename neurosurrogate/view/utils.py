from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..currents import CURRENT_MAP
from .engine import error_fig

if TYPE_CHECKING:
    from ..metrics.eval_sweep import CurrentSweepConfig


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
