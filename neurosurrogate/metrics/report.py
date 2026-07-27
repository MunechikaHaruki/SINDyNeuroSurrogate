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
回した結果でも) 描ける。描画の宣言は別ファイル (`draw.json`) の関心で、ここが型
(`DrawSpec`/`ResultSpec`/`ReportSpec`/`CompareSpec`) として持つ (計算仕様 `SimSpec`
を `eval.spec` が持つのと同じ関係)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from ..core.network import NeuronGraph
from ..eval.run import SimKey
from ..eval.spec import dedupe_labels
from ..eval.store import SimResult
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.diagnostics import preprocessed_latent, surrogate_metrics
from . import select
from .engine import error_fig
from .figs.cell import cell_figs, current_preview_fig
from .figs.grid import compare_grid_fig, metric_fig, trace_grid_fig
from .figs.model import closure_figs, neuron_graph_figs, preprocessor_figs
from .figs.train import train_figs
from .figs.wave import wave_report
from .save import SaveEntry, save_entries, slug
from .wave import dm_of

# --- 描画の宣言 (表示設定 + 並べ方) ---------------------------------------------


@dataclass(frozen=True, kw_only=True)
class DrawSpec:
    """表示設定のスキーマを知る唯一の場所 (UI は同名キーの dict を組むだけ = キーの
    意味と既定値の源はここ 1 つ)。**評価 (系列名) ごとに固有の値** (適用先が変われば
    comp 名も変わる) — グローバルな既定値を持たない。`plt_style` だけは評価に依らない
    ので `ReportSpec.plt_style` (トップレベル) の関心。"""

    eval_comp: str = ""  # 比較対象 comp (1 件)。適用先ごとに違うので既定を持たない
    view_comps: tuple[str, ...] = ()  # 全 comp を並べる図の表示制限 (空=全部)
    detail_point: int = 0  # 詳細図 (diff/attractor/指標) を描く点の index
    spike_orig: int = 0
    spike_surr: int = 0
    # 点軸メトリクス図 (点が 2 つ以上のときだけ描く折れ線)
    metric: str = "spike_count"
    metric_yauto: bool = True
    metric_ymin: float = 0.0
    metric_ymax: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> DrawSpec:
        """`draw.json` の `results[]` 1 件 → 型 (欠落キーは既定値)。
        **dict を見るのはここまで**で、以降は型で渡す。"""
        return cls(
            eval_comp=str(d.get("eval_comp") or ""),
            view_comps=tuple(str(c) for c in d.get("view_comps", ())),
            detail_point=int(d.get("detail_point", cls.detail_point)),
            spike_orig=int(d.get("spike_orig", cls.spike_orig)),
            spike_surr=int(d.get("spike_surr", cls.spike_surr)),
            metric=str(d.get("metric", cls.metric)),
            metric_yauto=bool(d.get("metric_yauto", cls.metric_yauto)),
            metric_ymin=float(d.get("metric_ymin", cls.metric_ymin)),
            metric_ymax=float(d.get("metric_ymax", cls.metric_ymax)),
        )

    def view_comp_ids(self, net: NeuronGraph) -> list[int] | None:
        """全 comp を並べる図に描く comp。UI では名前、描画側は comp_id で受ける。
        空選択 = 制限なし (None)。"""
        return [net.name_to_idx(c) for c in self.view_comps] or None

    def metric_ylim(self) -> tuple[float, float] | None:
        """点軸メトリクス図の y レンジ (auto なら None = matplotlib 任せ)。"""
        return None if self.metric_yauto else (self.metric_ymin, self.metric_ymax)


@dataclass(frozen=True, kw_only=True)
class CompareSpec:
    """複数の評価結果を 1 枚の格子へ縦に並べる図の宣言 (行=評価、列=点)。

    シミュ仕様ではなく**既に回した結果への参照**なので描画側の型 (compare を足しても
    回るシミュは eval entry を書いた分だけ)。`evals` は系列名 (`SimSpec.name`) 列。
    `eval_comp` は比較する compare 自身の設定 (系列ごとの `results[]` には
    紐付かないグローバル設定を持たないので、compare 自身が持つ)。
    """

    name: str
    evals: list[str]
    eval_comp: str

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            name=str(d["name"]),
            evals=[str(s) for s in d["evals"]],
            eval_comp=str(d["eval_comp"]),
        )


@dataclass(frozen=True, kw_only=True)
class ResultSpec:
    """`draw.json` の `results[]` 1 件 = 描く系列名 1 つ + その表示設定宣言。
    系列名ごとに固有の値 (comp 名など) しか持たないので、既定値からの override では
    なく `DrawSpec` 単体で完結する (欠落キーは `DrawSpec` の型既定値)。"""

    label: str
    draw: DrawSpec

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(label=str(d["eval"]), draw=DrawSpec.from_dict(d))


# `results`/`kinds` を指定しなければ手元の結果を全種描く既定 = 何も書かなくても
# 動く (最初の 1 回はここから)。絞り込みたくなったら明示的に列挙する。
ALL_KINDS = (
    "current",
    "summary",
    "closure",
    "preprocessor",
    "neurograph",
    "train",
    "traces",
    "cell",
    "metric",
    "compare",
)


@dataclass(frozen=True, kw_only=True)
class ReportSpec:
    """描画宣言全体 (`draw.json`) の型 = 表示スタイル + 系列名ごとの設定 + compare +
    保存する種類の絞り込み。**dict を見るのはここまで**で、以降
    (`model_report`/`eval_report`) は型で渡す (`eval.spec.parse_evals` が計算入力に
    対してやるのと同じ役目)。
    """

    plt_style: str = "presentation"
    results: tuple[ResultSpec, ...] = ()
    compares: dict[str, CompareSpec] = field(default_factory=dict)
    kinds: tuple[str, ...] = ALL_KINDS

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        default_d = d.get("default", {})
        results = [ResultSpec.from_dict(r) for r in d.get("results", ())]
        compares = [CompareSpec.from_dict(c) for c in d.get("compare", ())]
        kinds = tuple(str(k) for k in d["kinds"]) if "kinds" in d else ALL_KINDS
        return cls(
            plt_style=str(default_d.get("plt_style", cls.plt_style)),
            results=tuple(results),
            compares=dict(
                zip(dedupe_labels([c.name for c in compares]), compares, strict=True)
            ),
            kinds=kinds,
        )

    def draw_for(self, name: str) -> DrawSpec:
        """系列名の表示設定 (`results[]` に無ければ `DrawSpec` の型既定値。
        `eval_comp` が空だと `_eval_report_one` がエラー図を出す = 未指定は
        黙って何か描くのでなく気付ける形にする)。"""
        return next((r.draw for r in self.results if r.label == name), DrawSpec())

    def for_results(
        self, results: dict[SimKey, SimResult]
    ) -> list[tuple[str, DrawSpec]]:
        """描く対象 (系列名, draw) 列。`results` が空なら手元の結果を全部描く既定。
        非空なら **`results` に列挙した系列名だけへ絞り込む** (artifact が増えるほど
        draw 出力も比例して増えるのを避ける)。手元に無い名前は黙って落とす
        (`_compare_report` の「参照は宣言、欠落は宣言とのズレ」という扱いと揃える)。
        """
        names = select.series(results)
        if not self.results:
            return [(name, self.draw_for(name)) for name in names]
        return [(r.label, r.draw) for r in self.results if r.label in names]

    def wants(self, kind: str) -> bool:
        """この種類の図/表を保存するか (`kinds` による絞り込み)。"""
        return kind in self.kinds


# --- model (run のロードのみ。置換シミュ不要) -----------------------------------


def _summary_df(bundles: dict[str, SurrogateBundle]) -> pd.DataFrame:
    """run 軸の学習側指標サマリ (評価結果に依らないので results 無しでも出せる)。"""
    return pd.DataFrame(
        [{"label": label, **surrogate_metrics(s)} for label, s in bundles.items()]
    ).set_index("label")


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

    entries = (
        [
            SaveEntry(
                f"current/{name}", current_preview_fig(_first(name).spec.dataset())
            )
            for name, _ in targets
        ]
        if report.wants("current")
        else []
    )
    if not bundles:
        return entries
    bundle = next(iter(bundles.values()))
    nets = {_first(name).spec.target: _first(name).spec.net for name, _ in targets}
    # train データ図は適用先非依存 (学習データは meta から再生成)。comp 制限は
    # 代表系列 (先頭の results override) で名前解決 (学習 comp 名は target を
    # 跨いで共通)。
    comps = (
        targets[0][1].view_comp_ids(next(iter(nets.values())))
        if targets and nets
        else None
    )
    if report.wants("summary"):
        entries.append(SaveEntry("summary", _summary_df(bundles)))
    figs: list[tuple[str, Figure]] = []
    if report.wants("closure"):
        figs += closure_figs(bundle.closure)
    if report.wants("preprocessor"):
        figs += preprocessor_figs(bundle.preprocessor)
    if report.wants("neurograph"):
        figs += neuron_graph_figs(nets, bundle.meta)
    if report.wants("train"):
        figs += train_figs(bundle, comps)
    entries += [SaveEntry(name, fig) for name, fig in figs]
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
            *[SaveEntry(f"{name}/{run}/{fname}", fig) for fname, fig in figs],
            SaveEntry(f"{name}/{run}/metrics", rep.df_metrics),
            SaveEntry(f"{name}/{run}/metrics_scalar", rep.df_scalar),
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
    entries: list[SaveEntry] = []
    if report.wants("traces"):
        entries.append(
            SaveEntry(f"{name}/traces", trace_grid_fig(results, name, draw.eval_comp))
        )
    if report.wants("cell"):
        entries += _cell_entries(name, results, bundles, draw)
    if report.wants("metric") and len(labels) > 1:
        entries.append(
            SaveEntry(
                f"{name}/metric",
                metric_fig(
                    results, name, draw.eval_comp, draw.metric, draw.metric_ylim()
                ),
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
        fig = compare_grid_fig(results, spec.evals, spec.eval_comp)
        entries.append(SaveEntry(f"compare_{label}", fig))
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
    if report.wants("compare"):
        entries += _compare_report(report.compares, results)
    return entries


def render_report(
    bundles: dict[str, SurrogateBundle],
    results: dict[SimKey, SimResult],
    report: ReportSpec,
    draw_dict: dict,
    sources: list[str],
    dest: Path,
    style_paths: list[Path],
) -> list[Path]:
    """model/eval の図表を組み立てて dest へ保存する唯一の入口。呼び出し側
    (`scripts/marimo.py` の描画ボタン) は artifact 読込 + surrogate ロード (mlflow
    依存) だけを持ち、組立/保存はここに委譲する。"""
    for p in style_paths:
        plt.style.use(p)
    entries = model_report(bundles, results, report) + eval_report(
        results, bundles, report
    )
    meta = {"draw": draw_dict, "sources": sources}
    return save_entries(entries, dest, meta)
