"""**run 横断の図** = この選択でしか出ない成果物: 比べた N 本の
サマリ表、点を列・run を行に取る波形格子、点軸に沿った指標の折れ線。
marimo/MLflow 非依存。

**「点軸 × run 軸に開いた並び」を図/表に落とすのはここだけ** — 波形ドメイン
(`neurosurrogate.waveform`) は 1 ペア (原系, 置換系) しか知らず軸の話を持たない。
並び自体は `SeriesView` が既に持つので図の側で組み直さない。

`model` / `series` の図が run 1 本で決まる (別のレポートで見ても同じ) のに対し、
ここの図は「今 何本を比べているか」で中身が変わる = レポート run に属する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from ..core import access
from ..core.diverge import diverged
from ..plotting import Artifact, error_fig, new_figure, place_legend, use_style
from ..sim.catalog import currents
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.figures import summary_df
from ..waveform import dm_of, extract_metric
from . import SeriesView, run_names

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axes import Axes


# --- 点軸に沿った指標 -----------------------------------------------------------


def metrics_df(
    view: SeriesView,
    names: dict[str, str],
    comp_name: str,
    metric_key: str,
) -> pd.DataFrame:
    """系列の点軸に沿った metric の表 (列=run 軸、列名は `names` の run_id → 表示名)。
    原系の値は run に依らないので `original` 列 1 本へ畳む。"""
    comp_id = view.net.name_to_idx(comp_name)
    rows: list[dict] = []
    for index, orig in enumerate(view.points):
        row: dict = {"point": orig.point}
        for run_id in view.run_ids:
            o, s = view.pair(index, run_id)
            value, orig_value = extract_metric(dm_of(o, s, comp_id), metric_key)
            row[names[run_id]] = value
            if orig_value is not None:
                row["original"] = orig_value  # run に依らない = 同じ値の上書き
        rows.append(row)
    return pd.DataFrame(rows)


def metric_fig(
    view: SeriesView,
    names: dict[str, str],
    comp_name: str,
    metric_key: str,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """点軸に沿ったメトリクス折れ線 (Original + 各 run)。`names` は run_id → 表示名。
    marimo 非依存。"""
    data = metrics_df(view, names, comp_name, metric_key)
    axis = view.axis or "point"
    fig = new_figure()
    ax = fig.subplots()
    if "original" in data.columns:
        ax.plot(data["point"], data["original"], "k-o", label="Original", zorder=3)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, run_id in enumerate(view.run_ids):
        ax.plot(
            data["point"],
            data[names[run_id]],
            marker="s",
            linestyle="--",
            color=colors[idx % len(colors)],
            label=names[run_id],
        )

    ax.set_xlabel(axis)
    ax.set_ylabel(metric_key)
    ax.set_title(f"{axis} — {metric_key} ({comp_name})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    place_legend(ax)
    return fig


# --- 波形格子 (列=点、行=run) ---------------------------------------------------


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
    """波形格子の 1 行 = 列 (点) ごとの (原系, 重ねる置換系群) と対象 comp。
    行が run 軸か系列軸かの違いは、この列を組む側だけが知る。"""

    comp_id: int
    cells: list[tuple[xr.Dataset, dict[str, xr.Dataset]]]


def _grid_fig(
    header: list[tuple[float | None, xr.Dataset]],
    rows: list[_Row],
    axis_name: str | None,
) -> Figure:
    """波形格子の骨格 (列=点)。行1=I_ext (header の原系)、行2以降=`rows`。

    y レンジは行種別ごとに全列共有 (列間で高さを比べられる)。列数の揃わない行を
    混ぜると列の意味が行ごとにずれる → raise。点が 1 つ (掃引軸なし) なら列見出しも
    軸名も出さない = 単発が「1 列の格子」に素直に退化する。
    """
    n_col = len(header)
    if any(len(r.cells) != n_col for r in rows):
        raise ValueError("並べる結果は点数を揃える必要がある")
    # 高さは行数比例 + 固定オーバーヘッド。列見出し/x 軸ラベルは行数に
    # よらず一定高を占める → 比例分だけだと行数が少ないほど 1 行が潰れ、行数違い
    # の図を並べたとき波形の縦倍率が揃わない。
    fig = new_figure(figsize=(2.6 * n_col, 1.8 * len(rows) + 0.9))
    axes = fig.subplots(len(rows), n_col, squeeze=False, sharex=True)
    v_ylim = _shared_ylim(
        [access.potential(ds, r.comp_id) for r in rows for ds, _ in r.cells]
    )
    unit = currents.PARAM_UNITS.get(axis_name or "", "")

    for c, (value, _) in enumerate(header):
        if value is not None and axis_name:
            axes[0][c].set_title(f"{value:.3g} {unit}".strip())
    for r, row in enumerate(rows):
        for c, (orig_ds, surrs) in enumerate(row.cells):
            axes[r][c].set_ylim(*v_ylim)
            _trace_cell(axes[r][c], orig_ds, surrs, row.comp_id)
    for c in range(n_col):
        axes[-1][c].set_xlabel("t [ms]")
    place_legend(axes[0][-1])
    return fig


def _header(view: SeriesView) -> list[tuple[float | None, xr.Dataset]]:
    """列 (点) の見出しと I_ext 行に使う原系。"""
    return [(r.point, r.dataset) for r in view.points]


def _run_row(view: SeriesView, run_id: str, label: str, comp_id: int) -> _Row:
    """1 run を 1 行に開く (セルは原系 + その run 1 本)。"""
    return _Row(
        comp_id,
        [
            (orig.dataset, {label: surr.dataset})
            for orig, surr in zip(view.points, view.surrs[run_id], strict=True)
        ],
    )


def trace_grid_fig(view: SeriesView, names: dict[str, str], comp_name: str) -> Figure:
    """1 系列を run 軸で開いた波形格子 (行=run)。`names` は run_id → 表示名。"""
    comp_id = view.net.name_to_idx(comp_name)
    rows = [_run_row(view, rid, names[rid], comp_id) for rid in view.run_ids]
    return _grid_fig(_header(view), rows, view.axis)


# --- レポート 1 本の成果物 ------------------------------------------------------


def summary_figs(bundles: dict[str, SurrogateBundle]) -> list[Artifact]:
    """比べた N 本のサマリ表 (**由来は学習 run 群**だけ = 波形を読まない)。
    run 横断 = 中身が「今 何本を比べているか」で変わるのでレポートに属する。"""
    use_style()
    names = run_names(bundles)
    return summary_df({names[run_id]: bundle for run_id, bundle in bundles.items()})


def wave_report_figs(
    view: SeriesView,
    bundles: dict[str, SurrogateBundle],
    eval_comp: str,
    metric: str,
    metric_ylim: tuple[float, float] | None,
) -> list[Artifact]:
    """run 軸に開いた波形格子と点軸の折れ線 (**由来は読んだ波形 run**)。
    適用先に無い comp を指されたら図の代わりにエラー図 1 枚 (描画は止めない)。"""
    use_style()
    if eval_comp not in view.net.names:
        msg = f"{view.name}: eval_comp {eval_comp!r} not in {view.target!r}"
        return [Artifact("error", error_fig(msg))]
    names = run_names(bundles)
    figs = [Artifact("traces", trace_grid_fig(view, names, eval_comp))]
    if len(view.points) > 1:
        figs.append(
            Artifact("metric", metric_fig(view, names, eval_comp, metric, metric_ylim))
        )
    return figs
