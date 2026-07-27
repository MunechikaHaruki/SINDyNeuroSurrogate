"""**結果グリッド (`EvalGrid`) を軸で見る図**: 点軸に沿ったメトリクス折れ線と、
点を列に取る波形格子 2 種 (行=run / 行=評価 spec)。

格子の骨格は `_grid_fig` 1 本で、2 種の違いは**行の組み方だけ** (`_Row` 列を作る
side)。軸名も run 軸ラベルも結果 (`EvalGrid.spec` / `run_labels`) から引く =
呼び出し側で作り直さない。marimo 非依存。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ...core import access
from ..engine import new_figure, place_legend
from ..wave import diverged
from .wave import metrics_df

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axes import Axes

    from ...eval.eval import EvalGrid


def _axis_name(grid: EvalGrid) -> str | None:
    """点軸の名前 (掃引軸が無ければ None = 点が 1 つで軸を名乗らない)。"""
    return grid.spec.sweep.param if grid.spec.sweep else None


def metric_fig(
    grid: EvalGrid,
    comp_name: str,
    metric_key: str,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """点軸に沿ったメトリクス折れ線 (Original + 各 run)。marimo 非依存。
    run 軸ラベルも点軸名も結果から引く = 別引数で持ち回らない。"""
    data = metrics_df(grid, comp_name, metric_key)
    axis = _axis_name(grid) or "point"
    fig = new_figure()
    ax = fig.subplots()
    if "original" in data.columns:
        ax.plot(data["point"], data["original"], "k-o", label="Original", zorder=3)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, label in enumerate(grid.run_labels):
        ax.plot(
            data["point"],
            data[label],
            marker="s",
            linestyle="--",
            color=colors[idx % len(colors)],
            label=label,
        )

    ax.set_xlabel(axis)
    ax.set_ylabel(metric_key)
    ax.set_title(f"{axis} — {metric_key} ({comp_name})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    place_legend(ax)
    return fig


def _shared_ylim(series: list) -> tuple[float, float]:
    """波形格子の 1 行で共有する y レンジ (列間で高さを比べられる)。V 行は発散しない
    Original 電位から決め (ニューロン挙動を捉えるレンジ)、発散した置換系はこの
    レンジで頭打ちにする。I_ext 行も同じ規則で揃える (掃引点ごとに軸が伸縮しない)。"""
    lo = min(float(v.min()) for v in series)
    hi = max(float(v.max()) for v in series)
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def _trace_cell(
    ax: Axes, orig_ds: xr.Dataset, surrs: dict[str, xr.Dataset], comp_id: int
) -> None:
    """波形格子の 1 セル = 原系 (黒) に置換系を重ねる。置換系が 1 本なら赤破線、
    複数なら色分けして名前を凡例へ。発散した置換系は描かず (レンジを潰すため)、
    1 本も描けなければ "diverged" を出す。"""
    ax.plot(
        access.time(orig_ds),
        access.potential(orig_ds, comp_id),
        "k-",
        lw=0.7,
        label="Original",
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    drawn = 0
    for i, (label, ds) in enumerate(surrs.items()):
        v = access.potential(ds, comp_id)
        if diverged(v):
            continue
        ax.plot(
            access.time(ds),
            v,
            "--",
            lw=0.7,
            color="tab:red" if len(surrs) == 1 else colors[i % len(colors)],
            label="surrogate" if len(surrs) == 1 else label,
        )
        drawn += 1
    if drawn == 0:
        ax.text(
            0.5,
            0.5,
            "diverged",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="red",
        )


@dataclass(frozen=True)
class _Row:
    """波形格子の 1 行 = 行名 + 列 (点) ごとの (原系, 重ねる置換系群) と対象 comp。
    行が run 軸か spec 軸かの違いは、この列を組む側だけが知る。"""

    label: str
    comp_id: int
    cells: list[tuple[xr.Dataset, dict[str, xr.Dataset]]]


def _grid_fig(
    header: list[tuple[float | None, xr.Dataset]],
    rows: list[_Row],
    axis_name: str | None,
    comp_name: str,
) -> Figure:
    """波形格子の骨格 (列=点)。行1=I_ext (header の原系)、行2以降=`rows`。

    y レンジは行種別ごとに全列共有 (列間で高さを比べられる)。列数の揃わない行を
    混ぜると列の意味が行ごとにずれる → raise。点が 1 つ (掃引軸なし) なら列見出しも
    軸名も出さない = 単発が「1 列の格子」に素直に退化する。
    """
    n_col, n_row = len(header), 1 + len(rows)
    if any(len(r.cells) != n_col for r in rows):
        raise ValueError("並べる結果は点数を揃える必要がある")
    fig = new_figure(figsize=(2.6 * n_col, 1.8 * n_row))
    axes = fig.subplots(n_row, n_col, squeeze=False, sharex=True)
    i_ylim = _shared_ylim([access.i_ext_values(ds) for _, ds in header])
    v_ylim = _shared_ylim(
        [access.potential(ds, r.comp_id) for r in rows for ds, _ in r.cells]
    )

    for c, (value, orig_ds) in enumerate(header):
        axes[0][c].plot(*access.i_ext(orig_ds), lw=0.8, color="tab:gray")
        axes[0][c].set_ylim(*i_ylim)
        if value is not None and axis_name:
            axes[0][c].set_title(f"{axis_name}={value:.3g}")
    for r, row in enumerate(rows, start=1):
        axes[r][0].set_ylabel(row.label, fontsize="small")
        for c, (orig_ds, surrs) in enumerate(row.cells):
            axes[r][c].set_ylim(*v_ylim)
            _trace_cell(axes[r][c], orig_ds, surrs, row.comp_id)
    axes[0][0].set_ylabel("I_ext")
    for c in range(n_col):
        axes[-1][c].set_xlabel("t [ms]")
    place_legend(axes[1][-1])
    prefix = f"{axis_name} " if axis_name else ""
    fig.suptitle(f"{prefix}waveform ({comp_name})")
    return fig


def _header(grid: EvalGrid) -> list[tuple[float | None, xr.Dataset]]:
    """列 (点) の見出しと I_ext 行に使う原系。"""
    return [(p.value, p.original) for p in grid.points]


def trace_grid_fig(grid: EvalGrid, comp_name: str) -> Figure:
    """1 評価を run 軸で開いた波形格子 (行=run、セルはその run 1 本だけ重ねる)。
    行順は結果の run 軸 (`grid.run_labels`) そのもの。marimo 非依存。"""
    comp_id = grid.spec.net.name_to_idx(comp_name)
    rows = [
        _Row(
            label,
            comp_id,
            [(p.original, {label: p.surrogates[label]}) for p in grid.points],
        )
        for label in grid.run_labels
    ]
    return _grid_fig(_header(grid), rows, _axis_name(grid), comp_name)


def compare_grid_fig(grids: dict[str, EvalGrid], comp_name: str) -> Figure:
    """複数の評価を並べた波形格子 (行=評価、セルは親 run 1 本だけ)。

    同じ掃引を適用先 (刺激位置) 違いで並べる図なので、電流行は先頭評価のものを
    1 回だけ描く (点数が揃わなければ `_grid_fig` が raise)。run 軸は sweep 子を
    含めず**親 run (`run_labels[0]`) のみ** — 子まで重ねると比較の主眼 (刺激位置差)
    が run 差に埋もれる。
    """
    first = next(iter(grids.values()))
    rows = []
    for label, g in grids.items():
        parent = g.run_labels[0]
        rows.append(
            _Row(
                label,
                g.spec.net.name_to_idx(comp_name),
                [(p.original, {parent: p.surrogates[parent]}) for p in g.points],
            )
        )
    return _grid_fig(_header(first), rows, _axis_name(first), comp_name)
