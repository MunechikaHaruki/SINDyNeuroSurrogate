from __future__ import annotations

import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

_LEGEND_ROWS = 8  # 凡例 1 列あたりの最大項目数
_LEGEND_MAX_COLS = 3  # これを超える本数は名前で追えない → 凡例ごと省く


def new_figure(figsize: tuple[float, float] | None = None) -> Figure:
    """view の Figure はすべてここから作る。constrained layout が軸ラベル・凡例を
    含めて配置を解く → はみ出しが起きない (tight_layout は軸外の凡例を寸法計算に
    入れず figure の縁で切る)。"""
    return Figure(figsize=figsize, layout="constrained")


def place_legend(ax: Axes, handles: Sequence[Artist] | None = None) -> None:
    """凡例は必ず軸の外・右上へ。constrained layout が凡例の幅ぶん軸を縮めるので、
    波形に被らず figure 枠からも出ない。項目が列に収まらない図 (traub19 の
    comp×gate 等) は名前で判別できないので凡例自体を落とす。"""
    entries = handles if handles is not None else ax.get_legend_handles_labels()[0]
    if not entries:
        return
    ncols = math.ceil(len(entries) / _LEGEND_ROWS)
    if ncols > _LEGEND_MAX_COLS:
        return
    ax.legend(
        handles=entries,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize="small",
        frameon=False,
        ncols=ncols,
    )


def error_fig(msg: str) -> Figure:
    """描画失敗を赤テキストの Figure に畳む。戻り値型を fig で統一するため。
    失敗は握り潰さず標準エラー/ログにも流す (marimo 表示外でも気付けるように)。"""
    logger.error("描画失敗: %s", msg)
    print(f"[view] 描画失敗: {msg}", file=sys.stderr)
    fig = new_figure()
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
    fig = new_figure(figsize=figsize)
    axs = fig.subplots(nrows=n_rows, ncols=1, sharex=True)
    if n_rows == 1:
        axs = [axs]

    for ax, p in zip(axs, panels, strict=False):
        for tr in p.traces:
            x, y = tr.xy()
            ax.plot(x, y, label=tr.label, color=tr.color, linestyle=tr.style)
        ax.set_ylabel(p.ylabel)
        place_legend(ax)
        if p.xlabel:
            ax.set_xlabel(p.xlabel)

    return fig
