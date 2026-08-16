"""**結果 (`result.SeriesResults`) → 図**。marimo/MLflow 非依存で、返すのはどれも
`list[Artifact]` = **どの関数を呼んだかが保存段を決める** (図は属する run も由来も
名乗らない)。

図は run 横断か否かで 2 種:

- **run 横断** (`summary_figs` / `wave_report_figs`) … 中身が「今 何本を比べているか」
  で変わる = レポート run に属する。**「点軸 × run 軸に開いた並び」を図/表に落とすのは
  ここだけ** — 波形ドメイン (`neurosurrogate.waveform`) は 1 ペア (原系, 置換系) しか
  知らず軸の話を持たない。並び自体は `SeriesResults` が既に持つので図の側で組み直さない
- **波形 1 本で決まる** (`original_figs` / `detail_figs`) … 別のレポートで見ても同じ図
  (だから保存段も評価 run 側で、レポートを増やしても複製されない)。**単発と掃引で
  経路を分けない** — 点が 1 つでも点 index を名前に持つ 1 組が出るだけ

**学習 run 1 本の自己記述図はここに無い** (置換シミュの結果が要らない =
`surrogate.figures`)。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from ..core import access
from ..core.diverge import diverged
from ..plotting import Artifact, error_fig, new_figure, place_legend, use_style
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.diagnostics import preprocessed_latent
from ..surrogate.figures import summary_df
from ..waveform import (
    cell_figs,
    current_preview_fig,
    dm_of,
    extract_metric,
    wave_report,
)
from .catalog import currents
from .result import SeriesResults, SimResult

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


def _metrics_df(
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


def _metric_fig(
    view: SeriesResults,
    names: dict[str, str],
    comp_name: str,
    metric_key: str,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """点軸に沿ったメトリクス折れ線 (Original + 各 run)。`names` は run_id → 表示名。
    marimo 非依存。"""
    data = _metrics_df(view, names, comp_name, metric_key)
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
    # **軸まわりは列数/行数に依らず一定の幅と高さを食う**ので、波形に使う分
    # (列数/行数比例) と別に固定オーバーヘッドを足す。比例分だけだと、行見出し +
    # y 目盛 + 軸外の凡例で 1 列の格子は波形が数 mm まで潰れ、行数違いの図で波形の
    # 倍率も揃わない (constrained layout は figure を広げず軸を縮めて収めるため)。
    fig = new_figure(figsize=(2.6 * n_col + 2.6, 1.8 * n_row + 0.9))
    axes = fig.subplots(n_row, n_col, squeeze=False, sharex=True)
    ylim = _shared_ylim([access.potential(r.dataset, comp_id) for r in view.points])
    unit = currents.PARAM_UNITS.get(view.axis or "", "")

    for c, value in enumerate(view.values):
        if value is not None and view.axis:
            axes[0][c].set_title(f"{value:.3g} {unit}".strip())
    for r, run_id in enumerate(view.run_ids):
        # 行見出しは凡例用の複数行ラベル (`meta.label`) をそのまま使う = 凡例と同じ
        # 読み方で行を引ける。回転して置くので 1 行あたりの幅が効く → 小さめに。
        axes[r][0].set_ylabel(names[run_id], fontsize="small")
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
            Artifact("metric", _metric_fig(view, names, eval_comp, metric, metric_ylim))
        )
    return figs


# --- 波形 1 本で決まる図 --------------------------------------------------------


def original_figs(view: SeriesResults) -> list[Artifact]:
    """原系の波形 1 本だけで決まる図 (入力電流)。"""
    use_style()
    return [Artifact("current", current_preview_fig(view.points[0].spec))]


def detail_figs(
    view: SeriesResults,
    run_id: str,
    bundle: SurrogateBundle,
    eval_comp: str,
    view_comps: tuple[str, ...],
    detail_point: int,
    spike_orig: int,
    spike_surr: int,
) -> list[Artifact]:
    """選択した 1 点 × **1 モデル**の詳細図 + メトリクス表。描く対象は 1 つの置換系の
    波形そのもの (run 横断でない) = 同じ波形を別のレポートで見ても同じ図。点 index を
    名前に入れるので、つまみを動かしても前の点を上書きしない。

    潜在射影は run ごとの surrogate が要るので bundle を受け取る (結果 artifact は
    surrogate を持たない = 呼び出し側が run_id で対応付ける)。
    """
    use_style()
    net = view.net
    if eval_comp not in net.names:
        return []
    index = min(detail_point, len(view.points) - 1)  # 設定が点数を超えていても描く
    comp_id = net.name_to_idx(eval_comp)
    orig, surr = view.pair(index, run_id)
    cells = cell_figs(
        orig.dataset,
        surr.dataset,
        comp_id,
        lambda: preprocessed_latent(bundle, net, orig.dataset, comp_id),
        [net.name_to_idx(c) for c in view_comps] or None,
    )
    metrics = wave_report(dm_of(orig, surr, comp_id), spike_orig, spike_surr)
    # 点 index は名前の 1 段目 (保存段の下でそのまま階層になる)
    return [replace(a, name=f"p{index}/{a.name}") for a in (*cells, *metrics)]
