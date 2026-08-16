"""**run 横断の図** = この選択でしか出ない成果物: 比べた N 本の
サマリ表、点を列・run を行に取る波形格子、点軸に沿った指標の折れ線。
marimo/MLflow 非依存。

**「点軸 × run 軸に開いた並び」を図/表に落とすのはここだけ** — 波形ドメイン
(`neurosurrogate.waveform`) は 1 ペア (原系, 置換系) しか知らず軸の話を持たない。
並び自体は `SeriesResults` が既に持つので図の側で組み直さない。

`series` の図 (波形 1 本) や `surrogate.figures` の図 (学習 run 1 本) が run 1 本で
決まる (別のレポートで見ても同じ) のに対し、ここの図は「今 何本を比べているか」で
中身が変わる = レポート run に属する。
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from ...core import access
from ...core.diverge import diverged
from ...plotting import Artifact, error_fig, new_figure, place_legend, use_style
from ...surrogate.bundle import SurrogateBundle
from ...surrogate.figures import summary_df
from ...waveform import dm_of, extract_metric
from ..catalog import currents
from ..eval import SeriesResults, SimResult

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes


def run_names(bundles: dict[str, SurrogateBundle]) -> dict[str, str]:
    """run_id → 表示名 (凡例/行見出し)。**表示名は結果でなく surrogate 側から解く**
    (結果は run_id という同一性だけを持つ)。

    `meta.label` は学習構造 + 学習データまでしか区別しない → library_specs 違いや
    同 config の再実行は同じ label になるため、衝突したものにだけ与えた順の連番を
    付けて潰れを防ぐ (選択を拒否せず全部見せる)。**run 軸に何本並ぶか**を知らないと
    決まらないので、run 横断の図と同居する。
    """
    labels = [b.meta.label for b in bundles.values()]
    counts = Counter(labels)
    seen: Counter[str] = Counter()
    out: dict[str, str] = {}
    for run_id, label in zip(bundles, labels, strict=True):
        seen[label] += 1
        out[run_id] = label if counts[label] == 1 else f"{label}#{seen[label]}"
    return out


# --- 点軸に沿った指標 -----------------------------------------------------------


def metrics_df(
    view: SeriesResults,
    names: dict[str, str],
    comp_name: str,
    metric_key: str,
) -> pd.DataFrame:
    """系列の点軸に沿った metric の表 (列=run 軸、列名は `names` の run_id → 表示名)。
    原系の値は run に依らないので `original` 列 1 本へ畳む。"""
    comp_id = view.net.name_to_idx(comp_name)
    rows: list[dict[str, float | None]] = []
    for index, orig in enumerate(view.points):
        row: dict[str, float | None] = {"point": orig.point}
        for run_id in view.run_ids:
            o, s = view.pair(index, run_id)
            value, orig_value = extract_metric(dm_of(o, s, comp_id), metric_key)
            row[names[run_id]] = value
            if orig_value is not None:
                row["original"] = orig_value  # run に依らない = 同じ値の上書き
        rows.append(row)
    return pd.DataFrame(rows)


def metric_fig(
    view: SeriesResults,
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


def _shared_ylim(potentials: list[np.ndarray]) -> tuple[float, float]:
    """波形格子の全セルで共有する y レンジ (列間・行間で高さを比べられる)。発散しない
    Original 電位から決め (ニューロン挙動を捉えるレンジ)、発散した置換系はこの
    レンジで頭打ちにする。"""
    lo = min(float(v.min()) for v in potentials)
    hi = max(float(v.max()) for v in potentials)
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def _trace_cell(ax: Axes, orig: SimResult, surr: SimResult, comp_id: int) -> None:
    """波形格子の 1 セル = 原系 (黒) に置換系 1 本 (赤破線) を重ねる。発散した置換系は
    レンジを潰すので描かず "diverged" を出す。"""
    ax.plot(
        access.time(orig.dataset),
        access.potential(orig.dataset, comp_id),
        "k-",
        lw=0.7,
        label="Original",
    )
    v = access.potential(surr.dataset, comp_id)
    if diverged(v):
        ax.text(
            0.5,
            0.5,
            "diverged",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="red",
        )
        return
    ax.plot(
        access.time(surr.dataset), v, "--", lw=0.7, color="tab:red", label="surrogate"
    )


def trace_grid_fig(
    view: SeriesResults, names: dict[str, str], comp_name: str
) -> Figure:
    """1 系列を run 軸で開いた波形格子 (列=点、行=run。行見出しが run の表示名)。
    `names` は run_id → 表示名。

    列 (点) の数も行 (run) の数も `SeriesResults` が既に揃えているので、ここは並びを
    そのまま軸に落とすだけ。点が 1 つ (掃引軸なし) なら列見出しを出さない = 単発が
    「1 列の格子」に素直に退化する。
    """
    comp_id = view.net.name_to_idx(comp_name)
    n_col, n_row = len(view.points), len(view.run_ids)
    # 高さは行数比例 + 固定オーバーヘッド。列見出し/x 軸ラベルは行数によらず一定高を
    # 占める → 比例分だけだと行数が少ないほど 1 行が潰れ、行数違いの図を並べたとき
    # 波形の縦倍率が揃わない。
    fig = new_figure(figsize=(2.6 * n_col, 1.8 * n_row + 0.9))
    axes = fig.subplots(n_row, n_col, squeeze=False, sharex=True)
    ylim = _shared_ylim([access.potential(r.dataset, comp_id) for r in view.points])
    unit = currents.PARAM_UNITS.get(view.axis or "", "")

    for c, value in enumerate(view.values):
        if value is not None and view.axis:
            axes[0][c].set_title(f"{value:.3g} {unit}".strip())
    for r, run_id in enumerate(view.run_ids):
        axes[r][0].set_ylabel(names[run_id])
        for c in range(n_col):
            axes[r][c].set_ylim(*ylim)
            _trace_cell(axes[r][c], *view.pair(c, run_id), comp_id)
    for c in range(n_col):
        axes[-1][c].set_xlabel("t [ms]")
    place_legend(axes[0][-1])
    return fig


# --- レポート 1 本の成果物 ------------------------------------------------------


def summary_figs(bundles: dict[str, SurrogateBundle]) -> list[Artifact]:
    """比べた N 本のサマリ表 (**由来は学習 run 群**だけ = 波形を読まない)。
    run 横断 = 中身が「今 何本を比べているか」で変わるのでレポートに属する。"""
    use_style()
    names = run_names(bundles)
    return summary_df({names[run_id]: bundle for run_id, bundle in bundles.items()})


def wave_report_figs(
    view: SeriesResults,
    bundles: dict[str, SurrogateBundle],
    eval_comp: str,
    metric: str,
    metric_ylim: tuple[float, float] | None,
) -> list[Artifact]:
    """run 軸に開いた波形格子と点軸の折れ線 (**由来は読んだ波形 run**)。
    適用先に無い comp を指されたら図の代わりにエラー図 1 枚 (描画は止めない)。"""
    use_style()
    if eval_comp not in view.net.names:
        msg = f"eval_comp {eval_comp!r} not in {view.target!r}"
        return [Artifact("error", error_fig(msg))]
    names = run_names(bundles)
    figs = [Artifact("traces", trace_grid_fig(view, names, eval_comp))]
    if len(view.points) > 1:
        figs.append(
            Artifact("metric", metric_fig(view, names, eval_comp, metric, metric_ylim))
        )
    return figs
