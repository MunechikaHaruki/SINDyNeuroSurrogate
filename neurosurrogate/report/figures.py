"""**1 レポート = 1 系列 × N モデル**の図を集める層。marimo/MLflow 非依存。

`eval` が「何を回して何が出たか」を持つのに対し、ここは **どの図が出るか**。
レポートの単位は「ある系列の電流たちで N 本の surrogate を比べる」の 1 問 = 系列を
跨ぐ図は無い。

**返すのは `(どの run について描いたか, 名前, 中身)` の列だけ**で、保存先も保存名も
決めない (`ReportFig.kind`/`run_id` が「この図はどの run に属するか」というドメインの
事実で、それを段の名前に変えるのは MLflow を知る `scripts/artifacts.py`)。

**何を描くかを宣言しない**のが不変条件: モデル側は run 自身が描けるもの
(`surrogate_figs` が bundle の型から解く)、評価側は結果の形 (点が 2 つ以上なら
折れ線が出る) で決まる。**単発と掃引で経路を分けない** — 点が 1 つなら格子が 1 列に
なり点軸の折れ線が出ないだけ。

**描く対象は結果 (`SeriesView`) 自身**で、評価条件の宣言 (カタログの `SERIES`) は
受け取らない: 結果 artifact は入力仕様を自分で持つので、設定ファイルと無関係に
(別セッションで回した結果でも) 描ける。

**ドメインを横断する唯一の層**: 波形 (`neurosurrogate.waveform`) も surrogate の
自己記述 (`neurosurrogate.surrogate.figures`) も互いを知らず、両者を 1 つの報告へ
束ねるのはここだけ。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from matplotlib.figure import Figure

from ..core.network import NeuronGraph
from ..plotting import error_fig, use_style
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.diagnostics import preprocessed_latent
from ..surrogate.figures import summary_df, surrogate_figs
from ..waveform import cell_figs, current_preview_fig, dm_of, wave_report
from .grid import metric_fig, trace_grid_fig
from .results import SeriesView, run_names

# 図が属する run の種類。**保存段の名前ではない** (名前は MLflow を知る側が解く) —
# 「この図はどの run 1 本について描いたか」というドメインの事実。
MODEL = "model"  # 学習 run 1 本 (置換シミュの結果を受け取らない図)
ORIGINAL = "original"  # 原系の波形 1 本だけで決まる図
SURROGATE = "surrogate"  # 置換系の波形 1 本だけで決まる図
REPORT = "report"  # run 横断 = この選択でしか出ない図


@dataclass(frozen=True)
class Tuning:
    """**1 レポートの描画条件**。既定値と「どのキーがあるか」の単一源はここで、値を
    与える場所は `scripts/marimo.py` の widget 1 箇所だけ (カタログは「何を回すか」
    しか持たない = 描き方は図を見て決め直すものなので寿命が違う)。

    先頭の `eval_comp` だけ既定値が無い — 適用先が変われば comp 名も変わるので、
    系列を選んだ後に決まる。残りは既定のままでもレポートが出る。

    **何を描くかは宣言しない**: モデル側の図はその run が自分について描けるもの
    (`surrogate.figures.surrogate_figs` が bundle の型から解く)、評価側の図は結果の形
    (点が 2 つ以上なら折れ線が出る) で決まる。図の種類名がこの型に出てこないのが
    不変条件。
    """

    eval_comp: str  # 比較対象 comp (1 件)
    view_comps: tuple[str, ...] = ()  # 全 comp を並べる図の表示制限 (空=全部)
    metric: str = "spike_count"  # 点軸の折れ線に使う指標
    detail_point: int = 0  # 詳細図 (diff/attractor/指標) を描く点の index
    spike_orig: int = 0  # 特徴量比較に使う原系の何本目のスパイクか
    spike_surr: int = 0  # 同じく置換系
    metric_ylim: tuple[float, float] | None = None  # 折れ線の y レンジ (None=auto)


@dataclass(frozen=True)
class ReportFig:
    """1 成果物 = **どの run について描いたか** (`kind`, `run_id`) + 名前 + 中身。

    保存先も拡張子も持たない (`scripts/artifacts.SaveEntry` が段の名前を解いてから
    与える)。`sources` は**その図を描くのに読んだ run の id** で、どの experiment の
    id かは `kind` で決まる (モデル側の図は学習 run、結果側の図は評価 run)。手元の
    値をそのまま流すだけなので、その場で回した結果では評価 run の分が空になる。
    """

    kind: str  # MODEL / ORIGINAL / SURROGATE / REPORT
    run_id: str  # その kind の主体 (MODEL/SURROGATE は学習 run。他は空)
    name: str  # 図の名前 (`/` を含めば保存段の下でそのまま階層になる)
    obj: Figure | pd.DataFrame
    sources: tuple[str, ...] = ()


def _comp_ids(comps: tuple[str, ...], net: NeuronGraph) -> list[int] | None:
    """全 comp を並べる図に描く comp。宣言では名前、描画側は comp_id で受ける。
    空選択 = 制限なし (None)。"""
    return [net.name_to_idx(c) for c in comps] or None


# --- model (run のロードのみ。置換シミュ不要) -----------------------------------


def model_figs(bundles: dict[str, SurrogateBundle], tuning: Tuning) -> list[ReportFig]:
    """比べる N 本それぞれの自己記述図 (`kind=MODEL`)。

    **run 横断のサマリ表はここに無い** — 中身が「今 何本を比べているか」で変わるので
    `kind=REPORT` の産物 (レポートに属さないと別の選択で描き替わる)。

    描く対象は学習 run そのもの (置換シミュの結果を受け取らない)。適用先もbundleの
    学習datasetから解くため、系列やreport runを必要としない。

    **全 run 分描く** — レポートの単位が 1 系列 × N モデルなので N は比べたい本数
    そのもの (代表 1 本で済ませる必要がない)。学習データ図は `train_xr` の再生成を
    伴うが、それは N 本を比べると決めた分のコスト。
    """
    use_style()
    return [
        ReportFig(MODEL, run_id, name, fig, (run_id,))
        for run_id, bundle in bundles.items()
        for name, fig in surrogate_figs(
            bundle,
            bundle.meta.dataset.net,
            _comp_ids(tuning.view_comps, bundle.meta.dataset.net),
        )
    ]


# --- eval (系列の結果: 格子 + 選択点の詳細図 + 点軸メトリクス) ------------------


def _detail_figs(
    view: SeriesView, bundles: dict[str, SurrogateBundle], tuning: Tuning
) -> list[ReportFig]:
    """選択した 1 点 × 各モデルの詳細図 + メトリクス表 (`kind=SURROGATE`)。
    描く対象は 1 つの置換系の波形そのもの (run 横断でない) = 同じ波形を別のレポートで
    見ても同じ図。点 index を名前に入れるので、つまみを動かしても前の点を上書きしない。

    潜在射影は run ごとの surrogate が要るので bundles から引く (結果 artifact は
    surrogate を持たない = 描画側が run_id で対応付ける)。
    """
    index = view.clamp(tuning.detail_point)
    net = view.net
    comp_id = net.name_to_idx(tuning.eval_comp)
    figs: list[ReportFig] = []
    for run_id in view.run_ids:
        orig, surr = view.pair(index, run_id)
        cells = cell_figs(
            orig.dataset,
            surr.dataset,
            comp_id,
            # 潜在射影は原系だけで決まるが、どの surrogate で射影するかは run ごと
            # → run_id と原系をこの反復の値で束縛する (lazy 参照)。
            lambda rid=run_id, o=orig: preprocessed_latent(  # type: ignore[misc]
                bundles[rid], net, o.dataset, comp_id
            ),
            _comp_ids(tuning.view_comps, net),
        )
        metrics = wave_report(
            dm_of(orig, surr, comp_id), tuning.spike_orig, tuning.spike_surr
        )
        figs += [
            ReportFig(
                SURROGATE,
                run_id,
                f"p{index}/{name}",
                artifact,
                # 由来は原系と置換系の 2 本 (差分図はその対から出る)
                tuple(i for i in (view.original_id, view.series_id(run_id)) if i),
            )
            for name, artifact in (*cells, *metrics)
        ]
    return figs


def series_figs(
    view: SeriesView, bundles: dict[str, SurrogateBundle], tuning: Tuning
) -> list[ReportFig]:
    """評価run自身に属する入力電流とモデルごとの詳細成果物。"""
    use_style()
    if tuning.eval_comp not in view.net.names:
        return []
    return [
        ReportFig(
            ORIGINAL,
            "",
            "current",
            current_preview_fig(view.points[0].spec),
            tuple(i for i in (view.original_id,) if i),
        )
    ] + _detail_figs(view, bundles, tuning)


def report_figs(
    view: SeriesView, bundles: dict[str, SurrogateBundle], tuning: Tuning
) -> list[ReportFig]:
    """複数runを横断するレポート固有の成果物。"""
    use_style()
    if tuning.eval_comp not in view.net.names:
        msg = f"{view.name}: eval_comp {tuning.eval_comp!r} not in {view.target!r}"
        return [ReportFig(REPORT, "", "error", error_fig(msg))]
    names = run_names(bundles)
    figs = [
        ReportFig(REPORT, "", name, df, tuple(bundles))
        for name, df in summary_df(
            {names[run_id]: bundle for run_id, bundle in bundles.items()}
        )
    ]
    figs.append(
        ReportFig(
            REPORT,
            "",
            "traces",
            trace_grid_fig(view, names, tuning.eval_comp),
            view.sources,
        )
    )
    if len(view.points) > 1:
        figs.append(
            ReportFig(
                REPORT,
                "",
                "metric",
                metric_fig(
                    view, names, tuning.eval_comp, tuning.metric, tuning.metric_ylim
                ),
                view.sources,
            )
        )
    return figs
