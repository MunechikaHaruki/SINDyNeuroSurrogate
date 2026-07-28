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
回した結果でも) 描ける。描画宣言 (`draw.json`) は `parse_report` が正規化するだけで
以降も dict のまま持ち回る — `results[]`/`compare[]`/`kinds` は「表示にだけ使う
名前付き値の束」で、個々のキーへ都度アクセスするだけなので構造体化する意味が薄い。
キーの意味と既定値は `DEFAULT_DRAW`/`draw_for` が持つ。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt

from ..core.network import NeuronGraph
from ..eval.run import SimKey
from ..eval.store import SimResult, artifacts, load_all
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.diagnostics import preprocessed_latent
from . import select
from .artifact import (
    KIND_FUNCS,
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
from .save import SaveEntry, save_entries, slug


def parse_report(d: dict) -> dict:
    """`draw.json` の生 dict → 正規化した dict (`results`/`compares`/`kinds`)。
    `results[]`/`compare[]` は配列 → 名前キーの dict に畳み (`eval`/`name` キー
    自体は要素から取り除く)。以降 (`model_report`/`eval_report`) はこの dict を
    そのまま渡す (`eval.spec.parse_evals` が計算入力に対してやるのと同じ役目)。
    絞り込み判定 (`wants`/`draw_for`/`for_results`) はこのモジュールが持つ。
    """
    return {
        "results": {
            str(r["eval"]): {k: v for k, v in r.items() if k != "eval"}
            for r in d.get("results", ())
        },
        "compares": {
            str(c["name"]): {k: v for k, v in c.items() if k != "name"}
            for c in d.get("compare", ())
        },
        "kinds": {str(k): bool(v) for k, v in d.get("kinds", {}).items()},
    }


# `kinds` を指定しなければ手元の結果を全種描く既定 = 何も書かなくても動く (最初の
# 1 回はここから)。絞り込みたくなったら明示的に列挙する。キー名は `KIND_FUNCS`
# (`metrics/artifact/__init__.py`) の関数名から取る = 文字列を手で書き写さないので
# rename しても自動で追従する。
ALL_KINDS = tuple(f.__name__ for f in KIND_FUNCS)

# 表示設定 (`draw.json` の `results[]` 1 件) のキーと既定値の唯一の源。**評価
# (系列名) ごとに固有の値** (適用先が変われば comp 名も変わる) — グローバルな既定値
# を持たない (描画スタイルは呼び出し側 `scripts/marimo.py` の定数の関心で、ここには
# 持たない)。
DEFAULT_DRAW: dict = {
    "eval_comp": "",  # 比較対象 comp (1 件)。適用先ごとに違うので既定を持たない
    "view_comps": (),  # 全 comp を並べる図の表示制限 (空=全部)
    "detail_point": 0,  # 詳細図 (diff/attractor/指標) を描く点の index
    "spike_orig": 0,
    "spike_surr": 0,
    # 点軸メトリクス図 (点が 2 つ以上のときだけ描く折れ線)
    "metric": "spike_count",
    "metric_yauto": True,
    "metric_ymin": 0.0,
    "metric_ymax": 1.0,
}


def wants(report: dict, kind: str) -> bool:
    """この種類の図/表を保存するか (`report["kinds"]` による絞り込み)。未指定キーは
    描く既定 (`kinds` を明示指定したものだけが上書きされる)。"""
    return bool(report["kinds"].get(kind, True))


def draw_for(report: dict, name: str) -> dict:
    """系列名の表示設定 (`results[]` に無ければ `DEFAULT_DRAW` そのもの。指定した
    キーだけ既定を上書きする — `eval_comp` が空だと `_eval_report_one` がエラー図を
    出す = 未指定は黙って何か描くのでなく気付ける形にする)。"""
    return {**DEFAULT_DRAW, **report["results"].get(name, {})}


def for_results(
    report: dict, results: dict[SimKey, SimResult]
) -> list[tuple[str, dict]]:
    """描く対象 (系列名, draw) 列。`report["results"]` が空なら手元の結果を全部描く
    既定。非空なら **`results` に列挙した系列名だけへ絞り込む** (artifact が増える
    ほど draw 出力も比例して増えるのを避ける)。手元に無い名前は黙って落とす
    (`_compare_report` の「参照は宣言、欠落は宣言とのズレ」という扱いと揃える)。
    """
    names = select.series(results)
    if not report["results"]:
        return [(name, draw_for(report, name)) for name in names]
    return [
        (name, draw_for(report, name)) for name in names if name in report["results"]
    ]


def view_comp_ids(draw: dict, net: NeuronGraph) -> list[int] | None:
    """全 comp を並べる図に描く comp。UI では名前、描画側は comp_id で受ける。
    空選択 = 制限なし (None)。"""
    return [net.name_to_idx(c) for c in draw["view_comps"]] or None


def metric_ylim(draw: dict) -> tuple[float, float] | None:
    """点軸メトリクス図の y レンジ (auto なら None = matplotlib 任せ)。"""
    return None if draw["metric_yauto"] else (draw["metric_ymin"], draw["metric_ymax"])


# --- model (run のロードのみ。置換シミュ不要) -----------------------------------


def model_report(
    bundles: dict[str, SurrogateBundle],
    results: dict[SimKey, SimResult],
    report: dict,
) -> list[SaveEntry]:
    """静的モデル図 + 学習側サマリ表 + 電流プレビュー。`report["kinds"]` で種類ごとに
    出す/出さないを選べる (既定は全種類)。

    closure/preprocessor/train は**代表 run (先頭) のみ** — 全 run 分描くと学習
    データの再生成が run 数だけ走る (指標の run 横断比較は `summary` 表が担う)。
    neurograph は**結果の適用先ごと** (置換ノードが違う) = `for_results` の spec
    から引く (`eval_report` と同じ絞り込みに従う。計算入力の宣言
    `eval.json` はここでも見ない = 描画は結果だけを見るという不変条件を model 図にも
    適用する)。電流プレビューは回した入力そのものの確認用で系列ごとに 1 枚。
    """
    targets = for_results(report, results)

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
        if wants(report, "current_preview_fig")
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
        view_comp_ids(targets[0][1], next(iter(nets.values())))
        if targets and nets
        else None
    )
    if wants(report, "summary_df"):
        entries += [
            SaveEntry(name, df, sources=tuple(bundles.keys()))
            for name, df in summary_df(bundles)
        ]
    if wants(report, "closure_figs"):
        entries += [
            SaveEntry(name, fig, sources=(rep_run_id,))
            for name, fig in closure_figs(bundle.closure)
        ]
    if wants(report, "preprocessor_figs"):
        entries += [
            SaveEntry(name, fig, sources=(rep_run_id,))
            for name, fig in preprocessor_figs(bundle.preprocessor)
        ]
    if wants(report, "neuron_graph_figs"):
        target_sources = (s for name, _ in targets for s in _source_of(_first(name)))
        net_sources = tuple(dict.fromkeys((rep_run_id, *target_sources)))
        entries += [
            SaveEntry(name, fig, sources=net_sources)
            for name, fig in neuron_graph_figs(nets, bundle.meta)
        ]
    if wants(report, "train_figs"):
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
    draw: dict,
) -> list[SaveEntry]:
    """選択した 1 点 × 各 run の詳細図 + メトリクス df (名前は `<name>/<run>/...`)。

    潜在射影は run ごとの surrogate が要るので bundles から引く (結果 artifact は
    surrogate を持たない = 描画側が run_id で対応付ける)。
    """
    labels = select.labels_of(results, name)
    run_ids = select.run_ids_of(results, name)
    index = min(draw["detail_point"], len(labels) - 1)
    label = labels[index]
    orig = results[(label, None)]
    net = orig.spec.net
    comp_id = net.name_to_idx(draw["eval_comp"])
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
            view_comp_ids(draw, net),
        )
        metrics = wave_report(
            dm_of(orig, surr, comp_id), draw["spike_orig"], draw["spike_surr"]
        )
        # run 表示名は凡例向けに改行/`/` を含む → 名前に混ぜる分だけ slug 化
        run = slug(run_label)
        entries += [
            SaveEntry(f"{name}/{run}/{fname}", artifact, sources=sources, draw=draw)
            for fname, artifact in (*figs, *metrics)
        ]
    return entries


def _eval_report_one(
    name: str,
    results: dict[SimKey, SimResult],
    bundles: dict[str, SurrogateBundle],
    draw: dict,
    report: dict,
) -> list[SaveEntry]:
    """1 系列分: 波形格子 (点 × run) → 選択点の詳細図 → 点軸メトリクス折れ線。
    折れ線は**点が 2 つ以上のときだけ** (単発で 1 点の折れ線を出さない)。
    `report["kinds"]` で種類ごとに出す/出さないを選べる。"""
    labels = select.labels_of(results, name)
    net = results[(labels[0], None)].spec.net
    if draw["eval_comp"] not in net.names:
        # matplotlib テキストとして描かれる (CJK グリフ非対応) → 英語で書く。
        target = results[(labels[0], None)].spec.target
        eval_comp = draw["eval_comp"]
        msg = f"{name}: eval_comp {eval_comp!r} not in {target!r}"
        return [SaveEntry(f"{name}/error", error_fig(msg))]
    sources = select.sources_of(results, name)
    entries: list[SaveEntry] = []
    if wants(report, "trace_grid_fig"):
        entries.append(
            SaveEntry(
                f"{name}/traces",
                trace_grid_fig(results, name, draw["eval_comp"]),
                sources=sources,
                draw=draw,
            )
        )
    if wants(report, "cell_figs"):
        entries += _cell_entries(name, results, bundles, draw)
    if wants(report, "metric_fig") and len(labels) > 1:
        entries.append(
            SaveEntry(
                f"{name}/metric",
                metric_fig(
                    results, name, draw["eval_comp"], draw["metric"], metric_ylim(draw)
                ),
                sources=sources,
                draw=draw,
            )
        )
    return entries


def _compare_report(
    compares: dict[str, dict], results: dict[SimKey, SimResult]
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
        if any(s not in names for s in spec["evals"]):
            continue
        nets_ok = all(
            spec["eval_comp"]
            in results[(select.labels_of(results, s)[0], None)].spec.net.names
            for s in spec["evals"]
        )
        if not nets_ok:
            eval_comp = spec["eval_comp"]
            msg = f"{label}: some evals don't have eval_comp {eval_comp!r}"
            entries.append(SaveEntry(f"compare_{label}/error", error_fig(msg)))
            continue
        spec_sources = (s for n in spec["evals"] for s in select.sources_of(results, n))
        sources = tuple(dict.fromkeys(spec_sources))
        fig = compare_grid_fig(results, spec["evals"], spec["eval_comp"])
        entries.append(SaveEntry(f"compare_{label}", fig, sources=sources, draw=spec))
    return entries


def eval_report(
    results: dict[SimKey, SimResult],
    bundles: dict[str, SurrogateBundle],
    report: dict,
) -> list[SaveEntry]:
    """結果 (図 + メトリクス) → compare の格子図。

    描く対象は `for_results(report, results)` が決める: `draw.json` の `results` が
    空なら手元の結果を全部、非空ならそこに列挙した系列名だけ (計算入力の設定
    ファイル `eval.json` とは突き合わせない — artifact は別セッションの宣言で
    回したものでも描けるべき)。表示設定は `draw_for(report, name)`。
    """
    entries: list[SaveEntry] = []
    for name, draw in for_results(report, results):
        entries += _eval_report_one(name, results, bundles, draw, report)
    if wants(report, "compare_grid_fig"):
        entries += _compare_report(report["compares"], results)
    return entries


def render_report(
    bundles: dict[str, SurrogateBundle],
    results: dict[SimKey, SimResult],
    report: dict,
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


def load_and_render_report(
    draw_json: Path,
    artifact_dir: Path,
    sel_id: str | None,
    dest: Path,
    style_paths: list[Path],
    load_surrogate_model: Callable[[str], SurrogateBundle],
) -> list[Path]:
    """draw.json 読込から `render_report` までを一括した唯一の入口。呼び出し側
    (`scripts/marimo.py` の描画ボタン) は surrogate ロード (mlflow 依存) だけ
    `load_surrogate_model` として注入し、artifact 読込・組立・保存はここへ委譲する。
    """
    report = parse_report(json.loads(draw_json.read_text()))
    arts = artifacts(artifact_dir, sel_id)
    res = load_all(arts)
    bundles_for_draw = {
        a.meta.spec.run_id: load_surrogate_model(a.meta.spec.run_id)
        for a in arts
        if a.meta.spec.run_id is not None
    }
    return render_report(bundles_for_draw, res, report, dest, style_paths)
