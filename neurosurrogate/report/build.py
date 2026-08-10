"""評価結果 → 保存できる `SaveEntry` 列の組立、および `render_report` での保存まで。
marimo 非依存 (marimo は結果読込 + surrogate ロードだけ持ち、組立/保存はここへ
委譲する)。

`eval` が「何を回して何が出たか」を持つのに対し、ここは **どの図をどの名前で
並べるか**: model (置換シミュ不要の静的図 + 学習側サマリ) / eval (系列ごとの
波形格子・選択セルの詳細図・点軸メトリクス) の 2 グループを組み、呼び出し側は保存に
流すだけ。**単発と掃引で経路を分けない** — 点が 1 つなら格子が 1 列になり点軸の
折れ線が出ないだけ。

**描く対象は結果 (`ResultSet`) 自身**で、評価条件の宣言 (`eval.SERIES`) は
受け取らない: 結果 artifact は入力仕様を自分で持つので、設定ファイルと無関係に
(別セッションで回した結果でも) 描ける。描画宣言は `ReportSpec` (`.spec`) が型として
持つ = ここに文字列キーは出てこない。

**ドメインを横断する唯一の層**: 波形 (`neurosurrogate.waveform`) も surrogate の
自己記述 (`neurosurrogate.surrogate.figures`) も互いを知らず、両者を 1 つの報告へ
束ねるのはここだけ。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt

from ..core import access
from ..core.network import NeuronGraph
from ..plotting import error_fig
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.diagnostics import preprocessed_latent
from ..surrogate.figures import (
    closure_figs,
    neuron_graph_figs,
    preprocessor_figs,
    summary_df,
    train_figs,
)
from ..waveform import cell_figs, current_preview_fig, dm_of, wave_report
from .grid import compare_grid_fig, metric_fig, trace_grid_fig
from .results import ResultSet, SeriesView, run_names
from .save import SaveEntry, save_entries, slug
from .spec import CompareSpec, DrawSpec, ReportSpec


def _comp_ids(comps: tuple[str, ...], net: NeuronGraph) -> list[int] | None:
    """全 comp を並べる図に描く comp。宣言では名前、描画側は comp_id で受ける。
    空選択 = 制限なし (None)。"""
    return [net.name_to_idx(c) for c in comps] or None


def _i_ext_ylim(
    bundles: dict[str, SurrogateBundle], results: ResultSet
) -> tuple[float, float] | None:
    """train_raw.png と diff.png の I_ext パネルで揃える共通 y レンジ (発表用、
    5% パディング)。学習軌道 (train_xr) と評価結果の原系軌道すべてを見て決める。"""
    arrays = [access.i_ext_values(b.train_xr) for b in bundles.values()] + [
        access.i_ext_values(r.dataset) for view in results for r in view.points
    ]
    if not arrays:
        return None
    lo = min(float(a.min()) for a in arrays)
    hi = max(float(a.max()) for a in arrays)
    pad = (hi - lo) * 0.05 or 1.0
    return (lo - pad, hi + pad)


# --- model (run のロードのみ。置換シミュ不要) -----------------------------------


def model_report(
    bundles: dict[str, SurrogateBundle],
    results: ResultSet,
    report: ReportSpec,
    i_ext_ylim: tuple[float, float] | None = None,
) -> list[SaveEntry]:
    """静的モデル図 + 学習側サマリ表 + 電流プレビュー。`report.kinds` で種類ごとに
    出す/出さないを選べる (既定は全種類)。

    closure/preprocessor/train は**代表 run (先頭) のみ** — 全 run 分描くと学習
    データの再生成が run 数だけ走る (指標の run 横断比較は `summary` 表が担う)。
    neurograph は**結果の適用先ごと** (置換ノードが違う) = `report.targets` の
    view から引く (`eval_report` と同じ絞り込みに従う = 描画は結果だけを見るという
    不変条件を model 図にも適用する)。電流プレビューは回した入力そのものの確認用で
    系列ごとに 1 枚。
    """
    views = [(results[name], draw) for name, draw in report.targets(results.names)]

    entries = (
        [
            SaveEntry(
                f"current/{view.name}",
                current_preview_fig(view.points[0].spec.dataset()),
                sources=view.sources,
                draw=draw,
            )
            for view, draw in views
        ]
        if report.wants("current_preview_fig")
        else []
    )
    if not bundles:
        return entries
    rep_run_id, bundle = next(iter(bundles.items()))
    nets = {view.target: view.net for view, _ in views}
    # train データ図は適用先非依存 (学習データは meta から再生成)。comp 制限は
    # 代表系列 (先頭) で名前解決 (学習 comp 名は target を跨いで共通)。
    comps = _comp_ids(views[0][1].view_comps, views[0][0].net) if views else None
    if report.wants("summary_df"):
        entries += [
            SaveEntry(name, df, sources=tuple(bundles))
            for name, df in summary_df(bundles)
        ]
    if report.wants("closure_figs"):
        entries += [
            SaveEntry(name, fig, sources=(rep_run_id,))
            for name, fig in closure_figs(bundle.closure)
        ]
    if report.wants("preprocessor_figs"):
        entries += [
            SaveEntry(name, fig, sources=(rep_run_id,))
            for name, fig in preprocessor_figs(bundle.preprocessor)
        ]
    if report.wants("neuron_graph_figs"):
        net_sources = tuple(
            dict.fromkeys((rep_run_id, *(s for view, _ in views for s in view.sources)))
        )
        entries += [
            SaveEntry(name, fig, sources=net_sources)
            for name, fig in neuron_graph_figs(nets, bundle.meta)
        ]
    if report.wants("train_figs"):
        entries += [
            SaveEntry(name, fig, sources=(rep_run_id,))
            for name, fig in train_figs(bundle, comps, i_ext_ylim)
        ]
    return entries


# --- eval (系列ごとの結果: 詳細図 + 格子 + 点軸メトリクス) --------------


def _cell_entries(
    view: SeriesView,
    bundles: dict[str, SurrogateBundle],
    names: dict[str, str],
    draw: DrawSpec,
    i_ext_ylim: tuple[float, float] | None = None,
) -> list[SaveEntry]:
    """選択した 1 点 × 各 run の詳細図 + メトリクス df (名前は `<系列>/<run>/...`)。
    `names` は run_id → 表示名 (`results.run_names`)。

    潜在射影は run ごとの surrogate が要るので bundles から引く (結果 artifact は
    surrogate を持たない = 描画側が run_id で対応付ける)。
    """
    index = view.clamp(draw.detail_point)
    net = view.net
    comp_id = net.name_to_idx(draw.eval_comp)
    entries: list[SaveEntry] = []
    for run_id in view.run_ids:
        orig, surr = view.pair(index, run_id)
        figs = cell_figs(
            orig.dataset,
            surr.dataset,
            comp_id,
            # 潜在射影は原系だけで決まるが、どの surrogate で射影するかは run ごと
            # → run_id と原系をこの反復の値で束縛する (lazy 参照)。
            lambda rid=run_id, o=orig: preprocessed_latent(  # type: ignore[misc]
                bundles[rid], net, o.dataset, comp_id
            ),
            _comp_ids(draw.view_comps, net),
            i_ext_ylim,
        )
        metrics = wave_report(
            dm_of(orig, surr, comp_id), draw.spike_orig, draw.spike_surr
        )
        # run 表示名は凡例向けに改行/`/` を含む → 名前に混ぜる分だけ slug 化
        run = slug(names[run_id])
        entries += [
            SaveEntry(
                f"{view.name}/{run}/{fname}",
                artifact,
                sources=view.sources,
                draw=draw,
            )
            for fname, artifact in (*figs, *metrics)
        ]
    return entries


def _eval_report_one(
    view: SeriesView,
    bundles: dict[str, SurrogateBundle],
    names: dict[str, str],
    draw: DrawSpec,
    report: ReportSpec,
    i_ext_ylim: tuple[float, float] | None = None,
) -> list[SaveEntry]:
    """1 系列分: 波形格子 (点 × run) → 選択点の詳細図 → 点軸メトリクス折れ線。
    折れ線は**点が 2 つ以上のときだけ** (単発で 1 点の折れ線を出さない)。
    `report.kinds` で種類ごとに出す/出さないを選べる。"""
    if draw.eval_comp not in view.net.names:
        # matplotlib テキストとして描かれる (CJK グリフ非対応) → 英語で書く。
        msg = f"{view.name}: eval_comp {draw.eval_comp!r} not in {view.target!r}"
        return [SaveEntry(f"{view.name}/error", error_fig(msg))]
    entries: list[SaveEntry] = []
    if report.wants("trace_grid_fig"):
        entries.append(
            SaveEntry(
                f"{view.name}/traces",
                trace_grid_fig(view, names, draw.eval_comp),
                sources=view.sources,
                draw=draw,
            )
        )
    if report.wants("cell_figs"):
        entries += _cell_entries(view, bundles, names, draw, i_ext_ylim)
    if report.wants("metric_fig") and len(view.points) > 1:
        entries.append(
            SaveEntry(
                f"{view.name}/metric",
                metric_fig(view, names, draw.eval_comp, draw.metric, draw.metric_ylim),
                sources=view.sources,
                draw=draw,
            )
        )
    return entries


def _compare_report(
    compares: dict[str, CompareSpec],
    results: ResultSet,
    names: dict[str, str],
) -> list[SaveEntry]:
    """compare 宣言ごとの格子図 (行=系列、列=点)。compare 自身はシミュを増やさず、
    既に回した結果を並べるだけ。

    **参照先が手元に無い compare は黙って落とす** (error 図を出さない): 参照は設定側の
    宣言なので「まだ回していない / 別の結果を読んだ」は結果の欠陥ではなく宣言とのズレ
    (呼び出し側がログで扱う)。eval_comp の不一致だけは**手元の結果に対する表示設定の
    誤り**なので図に出す。
    """
    entries: list[SaveEntry] = []
    for label, spec in compares.items():
        if any(name not in results for name in spec.evals):
            continue
        views = [results[name] for name in spec.evals]
        if any(spec.eval_comp not in view.net.names for view in views):
            msg = f"{label}: some evals don't have eval_comp {spec.eval_comp!r}"
            entries.append(SaveEntry(f"compare_{label}/error", error_fig(msg)))
            continue
        sources = tuple(dict.fromkeys(s for view in views for s in view.sources))
        entries.append(
            SaveEntry(
                f"compare_{label}",
                compare_grid_fig(views, names, spec.eval_comp),
                sources=sources,
                draw=spec,
            )
        )
    return entries


def eval_report(
    results: ResultSet,
    bundles: dict[str, SurrogateBundle],
    report: ReportSpec,
    i_ext_ylim: tuple[float, float] | None = None,
) -> list[SaveEntry]:
    """結果 (図 + メトリクス) → compare の格子図。

    描く対象は `report.targets` が決める: `draw.json` の `results` が空なら手元の
    結果を全部、非空ならそこに列挙した系列名だけ (計算入力の宣言 `eval.SERIES` とは
    突き合わせない — 結果は別セッションの宣言で回したものでも描けるべき)。
    """
    names = run_names(bundles)
    entries: list[SaveEntry] = []
    for name, draw in report.targets(results.names):
        entries += _eval_report_one(
            results[name], bundles, names, draw, report, i_ext_ylim
        )
    if report.wants("compare_grid_fig"):
        entries += _compare_report(report.compares, results, names)
    return entries


def render_report(
    bundles: dict[str, SurrogateBundle],
    results: ResultSet,
    report: ReportSpec,
    dest: Path,
    style_paths: list[Path],
) -> list[Path]:
    """model/eval の図表を組み立てて dest へ保存する唯一の入口。呼び出し側
    (`scripts/marimo.py` の描画ボタン) は artifact 読込 + surrogate ロード (mlflow
    依存) だけを持ち、組立/保存はここに委譲する。成果物ごとの由来 (`sources`/`draw`)
    は各 `SaveEntry` が持ち、`meta.json` へは `save_entries` がそのまま落とす
    (draw.json 丸ごとの snapshot は持たない)。"""
    for p in style_paths:
        plt.style.use(p)
    # 学習側 (train_raw) は全軌道を覆うレンジ、評価側 (diff) は panels_diff の
    # 発表用既定に任せる (共有すると学習パルスの最大値で評価 step が潰れる)。
    entries = model_report(
        bundles, results, report, _i_ext_ylim(bundles, results)
    ) + eval_report(results, bundles, report)
    return save_entries(entries, dest)


def load_and_render_report(
    draw_json: Path,
    results: ResultSet,
    dest: Path,
    style_paths: list[Path],
    load_surrogate_model: Callable[[str], SurrogateBundle],
) -> list[Path]:
    """draw.json 読込から `render_report` までを一括した唯一の入口。呼び出し側
    (`scripts/marimo.py` の描画ボタン) は結果の読込と surrogate ロード (どちらも
    mlflow 依存) だけを持ち、宣言のパース・組立・保存はここへ委譲する。
    surrogate は結果に焼き込まれていない (run_id で対応付くだけ) ので、閉包項が
    要る図のために `load_surrogate_model` で引き直す。"""
    run_ids = dict.fromkeys(rid for view in results for rid in view.run_ids)
    bundles = {run_id: load_surrogate_model(run_id) for run_id in run_ids}
    return render_report(
        bundles, results, ReportSpec.load(draw_json), dest, style_paths
    )
