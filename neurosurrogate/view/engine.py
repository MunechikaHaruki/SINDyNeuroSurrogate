from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

import japanize_matplotlib  # noqa: F401  # rcParams を和文フォントへ (グローバル副作用)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

logger = logging.getLogger(__name__)

# rcParams 未適用経路でも和文が豆腐化しないよう明示指定する保険
_JP_FONT = FontProperties(fname=japanize_matplotlib.get_font_ttf_path())


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
        fontproperties=_JP_FONT,
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
) -> Figure:
    panels = [*spec[:-1], spec[-1].with_xlabel("Time [ms]")] if spec else spec

    n_rows = len(panels)
    fig = plt.figure()
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
