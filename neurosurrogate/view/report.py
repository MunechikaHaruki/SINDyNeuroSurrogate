"""評価結果 → 保存できる `SaveEntry` 列の組立。marimo 非依存 (CLI `scripts/draw.py`
からも同じ関数を呼ぶ)。

`metrics.eval` が「何を回して何が出たか」を持つのに対し、ここは **どの図をどの名前で
並べるか**: model (置換シミュ不要の静的図 + 学習側サマリ) / eval (結果グリッドごとの
波形格子・選択セルの詳細図・点軸メトリクス) の 2 グループを組み、呼び出し側は保存に
流すだけ。**単発と掃引で経路を分けない** — 点が 1 つなら格子が 1 列になり点軸の
折れ線が出ないだけ。

**描く対象は結果 `res` 自身**で、シミュ入力の設定 (`eval.json`) は受け取らない:
結果 artifact は入力仕様を自分で持つので、設定ファイルと無関係に (別セッションで
回した結果でも) 描ける。描画の宣言は別ファイル (`draw.json`) の関心で、ここが型
(`DrawSpec`/`ResultSpec`/`ReportSpec`/`CompareSpec`) として持つ (計算仕様 `EvalSpec`
を `metrics.spec` が持つのと同じ関係)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import pandas as pd

from ..core.network import NeuronGraph
from ..metrics.eval import EvalGrid, preprocessed_latent
from ..metrics.spec import dedupe_labels
from ..surrogate.bundle import SurrogateBundle
from .engine import error_fig
from .figs.cell import cell_figs, current_preview_fig
from .figs.grid import compare_grid_fig, metric_fig, trace_grid_fig
from .figs.model import closure_figs, neuron_graph_figs, preprocessor_figs
from .figs.train import train_figs
from .save import SaveEntry, slug

# --- 描画の宣言 (表示設定 + 並べ方) ---------------------------------------------


@dataclass(frozen=True, kw_only=True)
class DrawSpec:
    """表示設定のスキーマを知る唯一の場所 (UI は同名キーの dict を組むだけ = キーの
    意味と既定値の源はここ 1 つ)。"""

    plt_style: str = "presentation"
    eval_comp: str = ""  # 比較対象 comp (1 件)。既定は UI が適用先から決める
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
        """widget の値 / 保存 meta.json の `draw` セクション → 型 (欠落キーは既定値)。
        **dict を見るのはここまで**で、以降は型で渡す。"""
        return cls(
            plt_style=str(d.get("plt_style", cls.plt_style)),
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
        """全 comp を並べる図に描く comp。UI では名前、view 層は comp_id で受ける。
        空選択 = 制限なし (None)。"""
        return [net.name_to_idx(c) for c in self.view_comps] or None

    def metric_ylim(self) -> tuple[float, float] | None:
        """点軸メトリクス図の y レンジ (auto なら None = matplotlib 任せ)。"""
        return None if self.metric_yauto else (self.metric_ymin, self.metric_ymax)


@dataclass(frozen=True, kw_only=True)
class CompareSpec:
    """複数の評価結果を 1 枚の格子へ縦に並べる図の宣言 (行=評価、列=点)。

    シミュ仕様ではなく**既に回した結果への参照**なので描画側の型 (compare を足しても
    回るシミュは eval entry を書いた分だけ)。`evals` は結果 dict のキー列。
    """

    name: str
    evals: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(name=str(d["name"]), evals=[str(s) for s in d["evals"]])


@dataclass(frozen=True, kw_only=True)
class ResultSpec:
    """`draw.json` の `results[]` 1 件 = ある label だけ表示設定を変える override 宣言。
    override しないキーは `ReportSpec.default` を引き継ぐ (このマージが唯一の
    差分計算場所)。"""

    label: str
    draw: DrawSpec

    @classmethod
    def from_dict(cls, d: dict, default: dict) -> Self:
        return cls(label=str(d["eval"]), draw=DrawSpec.from_dict({**default, **d}))


@dataclass(frozen=True, kw_only=True)
class ReportSpec:
    """描画宣言全体 (`draw.json`) の型 = 既定表示設定 + label ごとの override +
    compare。**dict を見るのはここまで**で、以降 (`model_report`/`eval_report`) は
    型で渡す (`metrics.spec.parse_evals` が計算入力に対してやるのと同じ役目)。
    """

    default: DrawSpec
    results: tuple[ResultSpec, ...] = ()
    compares: dict[str, CompareSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        default_d = d.get("default", {})
        results = [ResultSpec.from_dict(r, default_d) for r in d.get("results", ())]
        compares = [CompareSpec.from_dict(c) for c in d.get("compare", ())]
        return cls(
            default=DrawSpec.from_dict(default_d),
            results=tuple(results),
            compares=dict(
                zip(dedupe_labels([c.name for c in compares]), compares, strict=True)
            ),
        )

    def draw_for(self, label: str) -> DrawSpec:
        """label の表示設定 (override 宣言が無ければ既定)。"""
        return next((r.draw for r in self.results if r.label == label), self.default)

    def for_results(
        self, res: dict[str, EvalGrid]
    ) -> list[tuple[str, EvalGrid, DrawSpec]]:
        """描く対象 (label, grid, draw) 列。**手元の結果 (`res`) が全て** — `results`
        は override の宣言であって絞り込みではない (宣言に無い label も既定設定で
        描く = 計算と描画が切れている、という既存の不変条件を維持する)。
        """
        return [(label, grid, self.draw_for(label)) for label, grid in res.items()]


# --- model (run のロードのみ。置換シミュ不要) -----------------------------------


def _summary_df(bundles: dict[str, SurrogateBundle]) -> pd.DataFrame:
    """run 軸の学習側指標サマリ (評価結果に依らないので res 無しでも出せる)。"""
    return pd.DataFrame(
        [{"label": label, **s.metrics()} for label, s in bundles.items()]
    ).set_index("label")


def model_report(
    bundles: dict[str, SurrogateBundle],
    res: dict[str, EvalGrid],
    report: ReportSpec,
) -> list[SaveEntry]:
    """静的モデル図 + 学習側サマリ表 + 電流プレビュー。

    closure/preprocessor/train は**代表 run (先頭) のみ** — 全 run 分描くと学習
    データの再生成が run 数だけ走る (指標の run 横断比較は `summary` 表が担う)。
    neurograph は**結果の適用先ごと** (置換ノードが違う) = 手元の結果 (`res`) の
    spec から引く (計算入力の宣言 `eval.json` はここでも見ない = 描画は結果だけを見る
    という不変条件を model 図にも適用する)。電流プレビューは回した入力そのものの
    確認用で label ごとに 1 枚。
    """
    entries = [
        SaveEntry(f"current/{label}", current_preview_fig(grid.spec.dataset()))
        for label, grid in res.items()
    ]
    if not bundles:
        return entries
    bundle = next(iter(bundles.values()))
    nets = {grid.spec.target: grid.spec.net for grid in res.values()}  # 適用先ごと1枚
    # train データ図は適用先非依存 (学習データは meta から再生成)。comp 制限は
    # 代表 target で名前解決 (学習 comp 名は target を跨いで共通)。
    comps = report.default.view_comp_ids(next(iter(nets.values()))) if nets else None
    return [
        SaveEntry("summary", _summary_df(bundles)),
        *entries,
        *[
            SaveEntry(name, fig)
            for name, fig in [
                *closure_figs(bundle.closure),
                *preprocessor_figs(bundle.preprocessor),
                *neuron_graph_figs(nets, bundle.meta),
                *train_figs(bundle, comps),
            ]
        ],
    ]


# --- eval (spec ごとの結果グリッド: 詳細図 + 格子 + 点軸メトリクス) --------------


def _cell_entries(
    label: str,
    grid: EvalGrid,
    bundles: dict[str, SurrogateBundle],
    draw: DrawSpec,
) -> list[SaveEntry]:
    """選択した 1 点 × 各 run の詳細図 + メトリクス df (名前は `<label>/<run>/...`)。

    潜在射影は run ごとの surrogate が要るので bundles から引く (結果 artifact は
    surrogate を持たない = 描画側が run_label で対応付ける)。
    """
    net = grid.spec.net
    comp_id = net.name_to_idx(draw.eval_comp)
    index = min(draw.detail_point, len(grid.points) - 1)
    point = grid.points[index]
    entries: list[SaveEntry] = []
    for run_label in grid.run_labels:
        figs = cell_figs(
            point.original,
            point.surrogates[run_label],
            comp_id,
            lambda rl=run_label: preprocessed_latent(  # type: ignore[misc]
                bundles[rl], net, point.original, comp_id
            ),
            draw.view_comp_ids(net),
        )
        rep = grid.wave_report(
            index, run_label, comp_id, draw.spike_orig, draw.spike_surr
        )
        # run 軸キーは凡例向けに改行/`/` を含む → 名前に混ぜる分だけ slug 化
        run = slug(run_label)
        entries += [
            *[SaveEntry(f"{label}/{run}/{name}", fig) for name, fig in figs],
            SaveEntry(f"{label}/{run}/metrics", rep.df_metrics),
            SaveEntry(f"{label}/{run}/metrics_scalar", rep.df_scalar),
        ]
    return entries


def _eval_report_one(
    label: str, grid: EvalGrid, bundles: dict[str, SurrogateBundle], draw: DrawSpec
) -> list[SaveEntry]:
    """1 spec 分: 波形格子 (点 × run) → 選択点の詳細図 → 点軸メトリクス折れ線。
    折れ線は**点が 2 つ以上のときだけ** (単発で 1 点の折れ線を出さない)。"""
    if draw.eval_comp not in grid.spec.net.names:
        # matplotlib テキストとして描かれる (CJK グリフ非対応) → 英語で書く。
        msg = f"{label}: eval_comp {draw.eval_comp!r} not in {grid.spec.target!r}"
        return [SaveEntry(f"{label}/error", error_fig(msg))]
    entries = [SaveEntry(f"{label}/traces", trace_grid_fig(grid, draw.eval_comp))]
    entries += _cell_entries(label, grid, bundles, draw)
    if grid.swept:
        entries.append(
            SaveEntry(
                f"{label}/metric",
                metric_fig(grid, draw.eval_comp, draw.metric, draw.metric_ylim()),
            )
        )
    return entries


def _compare_report(
    compares: dict[str, CompareSpec], res: dict[str, EvalGrid], draw: DrawSpec
) -> list[SaveEntry]:
    """compare spec ごとの格子図 (行=評価、列=点)。compare 自身はシミュを増やさず、
    既に回した結果を並べるだけ。

    **参照先が手元に無い compare は黙って落とす** (error 図を出さない): 参照は設定側の
    宣言なので「まだ回していない / 別の結果を読んだ」は結果の欠陥ではなく宣言とのズレ
    (呼び出し側がログで扱う)。ここで error 図にすると、結果を 1 本も持たない状態でも
    毎回赤い図が出て、計算と描画を切り離した意味が消える。eval_comp の不一致だけは
    **手元の結果に対する表示設定の誤り**なので図に出す。
    """
    eval_comp = draw.eval_comp
    entries: list[SaveEntry] = []
    for label, spec in compares.items():
        if any(s not in res for s in spec.evals):
            continue
        if not all(eval_comp in res[s].spec.net.names for s in spec.evals):
            msg = f"{label}: some evals don't have eval_comp {eval_comp!r}"
            entries.append(SaveEntry(f"compare_{label}/error", error_fig(msg)))
            continue
        fig = compare_grid_fig({s: res[s] for s in spec.evals}, eval_comp)
        entries.append(SaveEntry(f"compare_{label}", fig))
    return entries


def eval_report(
    res: dict[str, EvalGrid],
    bundles: dict[str, SurrogateBundle],
    report: ReportSpec,
) -> list[SaveEntry]:
    """**手元の結果を全部**描く (図 + メトリクス) → compare の格子図。

    描く対象は `res` のキーだけ = 計算入力の設定ファイル (`eval.json`) と突き合わせ
    ない: artifact を読んだ結果は別セッションの宣言で回したものでも描けるべきで、
    「宣言に無いから描かない」は計算と描画を再び結び付けてしまう。表示設定は
    `report.draw_for(label)` (override が無ければ既定 = `report.default`)。
    """
    entries: list[SaveEntry] = []
    for label, grid, draw in report.for_results(res):
        entries += _eval_report_one(label, grid, bundles, draw)
    return entries + _compare_report(report.compares, res, report.default)
