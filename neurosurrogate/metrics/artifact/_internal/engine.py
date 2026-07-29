"""描画プリミティブ: 図の生成・凡例配置・エラー図と、パネル記述 (`PanelSpec` /
`TraceSpec`) からの一括描画 (`draw_engine`)、複数図を (id, fig) 列へ畳む `collect`。

`figs/` 配下の各図はここだけを土台にする (matplotlib の作法をここへ閉じ込める)。
`TraceSpec` は t/y を numpy で持つので Dataset 非依存。marimo 非依存。
"""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Callable, Sequence
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
        fontsize="large",
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


def collect(jobs: dict[str, Callable[[], Figure]]) -> list[tuple[str, Figure]]:
    """名前付き描画 job を (id, fig) 列へ畳む — `figs/` が複数図を返すときの共通規約。
    1 図の失敗 (学習ドメイン外 comp 等) で列ごと落とさず error_fig に差し替える
    (呼び出し側は種別も成否も知らず保存/表示に流すだけ)。"""
    out: list[tuple[str, Figure]] = []
    for name, job in jobs.items():
        try:
            out.append((name, job()))
        except Exception as e:  # noqa: BLE001 — 描画の失敗は図に畳む (error_fig が記録)
            out.append((name, error_fig(f"{name}: {e}")))
    return out


@dataclass
class TraceSpec:
    t: np.ndarray
    y: np.ndarray
    label: str | None = None
    color: str | None = None
    style: str = "-"


@dataclass
class PanelSpec:
    ylabel: str
    traces: list[TraceSpec] = field(default_factory=list)
    ylim: tuple[float, float] | None = None


def draw_engine(
    spec: list[PanelSpec],
    figsize: tuple[float, float] | None = None,
) -> Figure:
    n_rows = len(spec)
    # figsize 未指定は matplotlib 既定。パネル数が多い図 (ゲート/潜在ごとに 1 段) は
    # 呼び出し側が段数に応じた寸法を渡す。
    fig = new_figure(figsize=figsize)
    axs = fig.subplots(nrows=n_rows, ncols=1, sharex=True)
    if n_rows == 1:
        axs = [axs]

    seen_labels: set[str] = set()
    for ax, p in zip(axs, spec, strict=False):
        for tr in p.traces:
            ax.plot(tr.t, tr.y, label=tr.label, color=tr.color, linestyle=tr.style)
        ax.set_ylabel(p.ylabel)
        if p.ylim is not None:
            ax.set_ylim(p.ylim)
        labels = {tr.label for tr in p.traces if tr.label is not None}
        if not labels or labels <= seen_labels:
            continue  # 全段同一 label 集合の反復 (comp 重ね) は凡例も反復せず 1 回のみ
        seen_labels |= labels
        place_legend(ax)
    if n_rows:  # 横軸は全段共有 (sharex) → ラベルは最下段だけ
        axs[-1].set_xlabel("Time [ms]")
    # 段ごとに y 目盛の桁数が違う → 既定では ylabel の x 位置が段ごとにずれる。
    fig.align_ylabels(axs)

    return fig
