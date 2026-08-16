"""**結果 (`result.SeriesResults`) → 単一の `Artifact`**。marimo/MLflow 非依存。

図は run 横断か否かで 2 種:

- **run 横断** (`summary_artifact` / `traces_artifact` / `metric_artifact`) …
  中身が「今 何本を比べているか」
  で変わる = レポート run に属する。**「点軸 × run 軸に開いた並び」を図/表に落とすのは
  ここだけ** — 波形ドメイン (`neurosurrogate.waveform`) は 1 ペア (原系, 置換系) しか
  知らず軸の話を持たない。並び自体は `SeriesResults` が既に持つので図の側で組み直さない
- **波形 1 本で決まる図**は `waveform.artifacts` が受け持つ

**学習 run 1 本の自己記述図はここに無い** (置換シミュの結果が要らない =
`surrogate.artifacts`)。成果物列への編成は `artifact.bundle` が受け持つ。
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from ..artifact.model import Artifact
from ..artifact.plotting import new_figure, place_legend
from ..core import access
from ..core.diverge import diverged
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.diagnostics import surrogate_metrics
from ..waveform.dynamics import DynamicMetrics, extract_metric
from .catalog import currents
from .result import SeriesResults

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
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
    for index, point in enumerate(view.values):
        row: dict[str, float | None] = {"point": point}
        for column in view.surrs:
            dm = DynamicMetrics(*view.pair(index, column), comp_id, view.dt)
            value, orig_value = extract_metric(dm, metric_key)
            row[names[str(column.run_id)]] = value
            if orig_value is not None:
                row["original"] = orig_value  # run に依らない = 同じ値の上書き
        rows.append(row)
    return pd.DataFrame(rows)


def metric_artifact(
    view: SeriesResults,
    names: dict[str, str],
    comp_name: str,
    metric_key: str,
    ylim: tuple[float, float] | None = None,
) -> Artifact:
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
    return Artifact("metric", fig)


# --- 波形格子 (列=点、行=run) ---------------------------------------------------


def _shared_ylim(potentials: list[np.ndarray]) -> tuple[float, float]:
    """波形格子の全セルで共有する y レンジ (列間・行間で高さを比べられる)。発散しない
    Original 電位から決め (ニューロン挙動を捉えるレンジ)、発散した置換系はこの
    レンジで頭打ちにする。"""
    lo = min(float(v.min()) for v in potentials)
    hi = max(float(v.max()) for v in potentials)
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def _trace_cell(ax: Axes, orig: xr.Dataset, surr: xr.Dataset, comp_id: int) -> None:
    """波形格子の 1 セル = 原系 (黒) に置換系 1 本 (赤破線) を重ねる。発散した置換系は
    レンジを潰すので描かず "diverged" を出す。"""
    ax.plot(
        access.time(orig),
        access.potential(orig, comp_id),
        "k-",
        lw=0.7,
        label="Original",
    )
    v = access.potential(surr, comp_id)
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
    ax.plot(access.time(surr), v, "--", lw=0.7, color="tab:red", label="surrogate")


def traces_artifact(
    view: SeriesResults, names: dict[str, str], comp_name: str
) -> Artifact:
    """1 系列を run 軸で開いた波形格子 (列=点、行=run。行見出しが run の表示名)。
    `names` は run_id → 表示名。

    列 (点) の数も行 (run) の数も `SeriesResults` が既に揃えているので、ここは並びを
    そのまま軸に落とすだけ。点が 1 つ (掃引軸なし) なら列見出しを出さない = 単発が
    「1 列の格子」に素直に退化する。
    """
    comp_id = view.net.name_to_idx(comp_name)
    n_col, n_row = len(view.points), len(view.surrs)
    # **軸まわりは列数/行数に依らず一定の幅と高さを食う**ので、波形に使う分
    # (列数/行数比例) と別に固定オーバーヘッドを足す。比例分だけだと、行見出し +
    # y 目盛 + 軸外の凡例で 1 列の格子は波形が数 mm まで潰れ、行数違いの図で波形の
    # 倍率も揃わない (constrained layout は figure を広げず軸を縮めて収めるため)。
    fig = new_figure(figsize=(2.6 * n_col + 2.6, 1.8 * n_row + 0.9))
    axes = fig.subplots(n_row, n_col, squeeze=False, sharex=True)
    ylim = _shared_ylim([access.potential(ds, comp_id) for ds in view.points])
    unit = currents.PARAM_UNITS.get(view.axis or "", "")

    for c, value in enumerate(view.values):
        if value is not None and view.axis:
            axes[0][c].set_title(f"{value:.3g} {unit}".strip())
    for r, column in enumerate(view.surrs):
        # 行見出しは凡例用の複数行ラベル (`meta.label`) をそのまま使う = 凡例と同じ
        # 読み方で行を引ける。回転して置くので 1 行あたりの幅が効く → 小さめに。
        axes[r][0].set_ylabel(names[str(column.run_id)], fontsize="small")
        for c in range(n_col):
            axes[r][c].set_ylim(*ylim)
            _trace_cell(axes[r][c], *view.pair(c, column), comp_id)
    for c in range(n_col):
        axes[-1][c].set_xlabel("t [ms]")
    place_legend(axes[0][-1])
    return Artifact("traces", fig)


# --- レポート 1 本の成果物 ------------------------------------------------------


def summary_artifact(bundles: dict[str, SurrogateBundle]) -> Artifact:
    """比べた N 本のサマリ表 (**由来は学習 run 群**だけ = 波形を読まない)。
    run 横断 = 中身が「今 何本を比べているか」で変わるのでレポートに属する。"""
    names = run_names(bundles)
    return Artifact(
        "summary",
        pd.DataFrame(
            [
                {"label": names[run_id], **surrogate_metrics(bundle)}
                for run_id, bundle in bundles.items()
            ]
        ).set_index("label"),
    )
