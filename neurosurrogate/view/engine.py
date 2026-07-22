from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def error_fig(msg: str) -> Figure:
    """描画失敗を赤テキストの Figure に畳む。戻り値型を fig で統一するため。
    失敗は握り潰さず標準エラー/ログにも流す (marimo 表示外でも気付けるように)。"""
    logger.error("描画失敗: %s", msg)
    print(f"[view] 描画失敗: {msg}", file=sys.stderr)
    fig = plt.figure()
    ax = fig.subplots()
    ax.text(
        0.5,
        0.5,
        msg,
        transform=ax.transAxes,
        ha="center",
        color="red",
        wrap=True,
    )
    ax.axis("off")
    return fig


@dataclass
class TraceSpec:
    t: np.ndarray
    y: np.ndarray
    label: str | None = None
    color: str | None = None
    style: str = "-"

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.t, self.y


@dataclass
class PanelSpec:
    ylabel: str
    traces: list[TraceSpec] = field(default_factory=list)
    xlabel: str | None = None

    def has_legend(self) -> bool:
        return any(tr.label for tr in self.traces)

    def with_xlabel(self, label: str) -> PanelSpec:
        return PanelSpec(ylabel=self.ylabel, traces=self.traces, xlabel=label)


def draw_engine(
    spec: list[PanelSpec],
    figsize: tuple[float, float] | None = None,
) -> Figure:
    panels = [*spec[:-1], spec[-1].with_xlabel("Time [ms]")] if spec else spec

    n_rows = len(panels)
    # figsize 未指定は matplotlib 既定。パネル数が多い図 (ゲート/潜在ごとに 1 段) は
    # 呼び出し側が段数に応じた寸法を渡す。
    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(nrows=n_rows, ncols=1, sharex=True)
    if n_rows == 1:
        axs = [axs]

    for ax, p in zip(axs, panels, strict=False):
        for tr in p.traces:
            x, y = tr.xy()
            ax.plot(x, y, label=tr.label, color=tr.color, linestyle=tr.style)
        ax.set_ylabel(p.ylabel)
        if p.has_legend():
            ax.legend()
        if p.xlabel:
            ax.set_xlabel(p.xlabel)

    fig.tight_layout()
    return fig
