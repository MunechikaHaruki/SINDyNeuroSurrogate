"""評価結果 → 保存できる `SaveEntry` 列の組立、および `render_report` での保存まで。
marimo 非依存 (marimo は artifact 読込 + surrogate ロードだけ持ち、組立/保存は
ここに委譲する)。

`eval.run` が「何を回して何が出たか」を持つのに対し、ここは **どの図をどの名前で
並べるか**: model (置換シミュ不要の静的図 + 学習側サマリ) / eval (系列ごとの
波形格子・選択セルの詳細図・点軸メトリクス) の 2 グループを組み、呼び出し側は保存に
流すだけ。**単発と掃引で経路を分けない** — 点が 1 つなら格子が 1 列になり点軸の
折れ線が出ないだけ。

**描く対象は結果 `results` 自身**で、シミュ入力の設定 (`eval.json`) は受け取らない:
結果 artifact は入力仕様を自分で持つので、設定ファイルと無関係に (別セッションで
回した結果でも) 描ける。描画の宣言のスキーマ (`DrawSpec`/`ReportSpec`/`CompareSpec`)
は `declare.py` が持つ (計算仕様 `SimSpec` を `eval.spec` が持つのと同じ関係、ここは
型を受け取って組み立てるだけ)。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..eval.run import SimKey
from ..eval.store import SimResult
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.diagnostics import preprocessed_latent
from . import select
from .artifact import (
    cell_figs,
    closure_figs,
    compare_grid_fig,
    current_preview_fig,
    error_fig,
    metric_fig,
    neuron_graph_figs,
    preprocessor_figs,
    summary_df,
    trace_grid_fig,
    train_figs,
    wave_report,
)
from .artifact._internal.wave import dm_of
from .declare import CompareSpec, DrawSpec, ReportSpec
from .save import SaveEntry, save_entries, slug

# --- model (run のロードのみ。置換シミュ不要) -----------------------------------


def model_report(
    bundles: dict[str, SurrogateBundle],
    results: dict[SimKey, SimResult],
    report: ReportSpec,
) -> list[SaveEntry]:
    """静的モデル図 + 学習側サマリ表 + 電流プレビュー。`report.kinds` で種類ごとに
    出す/出さないを選べる (既定は全種類)。

    closure/preprocessor/train は**代表 run (先頭) のみ** — 全 run 分描くと学習
    データの再生成が run 数だけ走る (指標の run 横断比較は `summary` 表が担う)。
    neurograph は**結果の適用先ごと** (置換ノードが違う) = `report.for_results(results)`
    の spec から引く (`eval_report` と同じ絞り込みに従う。計算入力の宣言
    `eval.json` はここでも見ない = 描画は結果だけを見るという不変条件を model 図にも
    適用する)。電流プレビューは回した入力そのものの確認用で系列ごとに 1 枚。
    """
    targets = report.for_results(results)

    def _first(name: str) -> SimResult:
        label = select.labels_of(results, name)[0]
        return results[(label, None)]

    def _source_of(r: SimResult) -> tuple[str, ...]:
        return (str(r.source),) if r.source is not None else ()

    entries = (
        [
            SaveEntry(
                f"current/{name}",
                current_preview_fig(_first(name).spec.dataset()),
                sources=_source_of(_first(name)),
                draw=draw,
            )
            for name, draw in targets
        ]
        if report.wants("current_preview_fig")
        else []
    )
    if not bundles:
        return entries
    rep_run_id = next(iter(bundles.keys()))
    bundle = bundles[rep_run_id]
    nets = {_first(name).spec.target: _first(name).spec.net for name, _ in targets}
    # train データ図は適用先非依存 (学習データは meta から再生成)。comp 制限は
    # 代表系列 (先頭の results override) で名前解決 (学習 comp 名は target を
    # 跨いで共通)。
    comps = (
        targets[0][1].view_comp_ids(next(iter(nets.values())))
        if targets and nets
        else None
    )
    if report.wants("summary_df"):
        entries.append(
            SaveEntry("summary", summary_df(bundles), sources=tuple(bundles.keys()))
        )
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
        target_sources = (s for name, _ in targets for s in _source_of(_first(name)))
        net_sources = tuple(dict.fromkeys((rep_run_id, *target_sources)))
        entries += [
            SaveEntry(name, fig, sources=net_sources)
            for name, fig in neuron_graph_figs(nets, bundle.meta)
        ]
    if report.wants("train_figs"):
        entries += [
            SaveEntry(name, fig, sources=(rep_run_id,))
            for name, fig in train_figs(bundle, comps)
        ]
    return entries


# --- eval (系列ごとの結果: 詳細図 + 格子 + 点軸メトリクス) --------------


def _cell_entries(
    name: str,
    results: dict[SimKey, SimResult],
    bundles: dict[str, SurrogateBundle],
    draw: DrawSpec,
) -> list[SaveEntry]:
    """選択した 1 点 × 各 run の詳細図 + メトリクス df (名前は `<name>/<run>/...`)。

    潜在射影は run ごとの surrogate が要るので bundles から引く (結果 artifact は
    surrogate を持たない = 描画側が run_id で対応付ける)。
    """
    labels = select.labels_of(results, name)
    run_ids = select.run_ids_of(results, name)
    index = min(draw.detail_point, len(labels) - 1)
    label = labels[index]
    orig = results[(label, None)]
    net = orig.spec.net
    comp_id = net.name_to_idx(draw.eval_comp)
    entries: list[SaveEntry] = []
    for run_id in run_ids:
        surr = results[(label, run_id)]
        run_label = select.run_label_of(results, name, run_id)
        sources = tuple(str(r.source) for r in (orig, surr) if r.source is not None)
        figs = cell_figs(
            orig.dataset,
            surr.dataset,
            comp_id,
            lambda rid=run_id: preprocessed_latent(  # type: ignore[misc]
                bundles[rid], net, orig.dataset, comp_id
            ),
            draw.view_comp_ids(net),
        )
        rep = wave_report(dm_of(orig, surr, comp_id), draw.spike_orig, draw.spike_surr)
        # run 表示名は凡例向けに改行/`/` を含む → 名前に混ぜる分だけ slug 化
        run = slug(run_label)
        entries += [
            *[
                SaveEntry(f"{name}/{run}/{fname}", fig, sources=sources, draw=draw)
                for fname, fig in figs
            ],
            SaveEntry(
                f"{name}/{run}/metrics",
                rep.df_metrics,
                sources=sources,
                draw=draw,
            ),
            SaveEntry(
                f"{name}/{run}/metrics_scalar",
                rep.df_scalar,
                sources=sources,
                draw=draw,
            ),
        ]
    return entries


def _eval_report_one(
    name: str,
    results: dict[SimKey, SimResult],
    bundles: dict[str, SurrogateBundle],
    draw: DrawSpec,
    report: ReportSpec,
) -> list[SaveEntry]:
    """1 系列分: 波形格子 (点 × run) → 選択点の詳細図 → 点軸メトリクス折れ線。
    折れ線は**点が 2 つ以上のときだけ** (単発で 1 点の折れ線を出さない)。
    `report.kinds` で種類ごとに出す/出さないを選べる。"""
    labels = select.labels_of(results, name)
    net = results[(labels[0], None)].spec.net
    if draw.eval_comp not in net.names:
        # matplotlib テキストとして描かれる (CJK グリフ非対応) → 英語で書く。
        target = results[(labels[0], None)].spec.target
        msg = f"{name}: eval_comp {draw.eval_comp!r} not in {target!r}"
        return [SaveEntry(f"{name}/error", error_fig(msg))]
    sources = select.sources_of(results, name)
    entries: list[SaveEntry] = []
    if report.wants("trace_grid_fig"):
        entries.append(
            SaveEntry(
                f"{name}/traces",
                trace_grid_fig(results, name, draw.eval_comp),
                sources=sources,
                draw=draw,
            )
        )
    if report.wants("cell_figs"):
        entries += _cell_entries(name, results, bundles, draw)
    if report.wants("metric_fig") and len(labels) > 1:
        entries.append(
            SaveEntry(
                f"{name}/metric",
                metric_fig(
                    results, name, draw.eval_comp, draw.metric, draw.metric_ylim()
                ),
                sources=sources,
                draw=draw,
            )
        )
    return entries


def _compare_report(
    compares: dict[str, CompareSpec], results: dict[SimKey, SimResult]
) -> list[SaveEntry]:
    """compare spec ごとの格子図 (行=系列、列=点)。compare 自身はシミュを増やさず、
    既に回した結果を並べるだけ。

    **参照先が手元に無い compare は黙って落とす** (error 図を出さない): 参照は設定側の
    宣言なので「まだ回していない / 別の結果を読んだ」は結果の欠陥ではなく宣言とのズレ
    (呼び出し側がログで扱う)。eval_comp の不一致だけは**手元の結果に対する表示設定の
    誤り**なので図に出す。
    """
    names = set(select.series(results))
    entries: list[SaveEntry] = []
    for label, spec in compares.items():
        if any(s not in names for s in spec.evals):
            continue
        nets_ok = all(
            spec.eval_comp
            in results[(select.labels_of(results, s)[0], None)].spec.net.names
            for s in spec.evals
        )
        if not nets_ok:
            msg = f"{label}: some evals don't have eval_comp {spec.eval_comp!r}"
            entries.append(SaveEntry(f"compare_{label}/error", error_fig(msg)))
            continue
        spec_sources = (s for n in spec.evals for s in select.sources_of(results, n))
        sources = tuple(dict.fromkeys(spec_sources))
        fig = compare_grid_fig(results, spec.evals, spec.eval_comp)
        entries.append(SaveEntry(f"compare_{label}", fig, sources=sources, draw=spec))
    return entries


def eval_report(
    results: dict[SimKey, SimResult],
    bundles: dict[str, SurrogateBundle],
    report: ReportSpec,
) -> list[SaveEntry]:
    """結果 (図 + メトリクス) → compare の格子図。

    描く対象は `report.for_results(results)` が決める: `draw.json` の `results` が
    空なら手元の結果を全部、非空ならそこに列挙した系列名だけ (計算入力の設定
    ファイル `eval.json` とは突き合わせない — artifact は別セッションの宣言で
    回したものでも描けるべき)。表示設定は `report.draw_for(name)`。
    """
    entries: list[SaveEntry] = []
    for name, draw in report.for_results(results):
        entries += _eval_report_one(name, results, bundles, draw, report)
    if report.wants("compare_grid_fig"):
        entries += _compare_report(report.compares, results)
    return entries


def render_report(
    bundles: dict[str, SurrogateBundle],
    results: dict[SimKey, SimResult],
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
    entries = model_report(bundles, results, report) + eval_report(
        results, bundles, report
    )
    return save_entries(entries, dest)
