"""**1 レポート = 1 系列 × N モデル**の組立と保存。marimo 非依存 (marimo は結果読込
+ surrogate ロードだけ持ち、組立/保存はここへ委譲する)。

`eval` が「何を回して何が出たか」を持つのに対し、ここは **どの図をどの名前で
並べるか**。レポートの単位は「ある系列の電流たちで N 本の surrogate を比べる」の 1 問
= 系列を跨ぐ図は無く、`dest` がそのまま 1 レポートの root になる。

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

from collections.abc import Callable
from pathlib import Path

from ..core import access
from ..core.network import NeuronGraph
from ..plotting import error_fig, use_style
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.diagnostics import preprocessed_latent
from ..surrogate.figures import summary_df, surrogate_figs
from ..waveform import cell_figs, current_preview_fig, dm_of, wave_report
from .grid import metric_fig, trace_grid_fig
from .results import SeriesView, run_names
from .save import SaveEntry, save_entries, slug
from .spec import NO_TUNING, Report, Tuning


def _comp_ids(comps: tuple[str, ...], net: NeuronGraph) -> list[int] | None:
    """全 comp を並べる図に描く comp。宣言では名前、描画側は comp_id で受ける。
    空選択 = 制限なし (None)。"""
    return [net.name_to_idx(c) for c in comps] or None


def _i_ext_ylim(
    bundles: dict[str, SurrogateBundle], view: SeriesView
) -> tuple[float, float] | None:
    """train_raw.png の I_ext パネルで揃える共通 y レンジ (発表用、5% パディング)。
    レポート内の学習軌道 (train_xr) と系列の原系軌道すべてを見て決める。"""
    arrays = [access.i_ext_values(b.train_xr) for b in bundles.values()] + [
        access.i_ext_values(r.dataset) for r in view.points
    ]
    if not arrays:
        return None
    lo = min(float(a.min()) for a in arrays)
    hi = max(float(a.max()) for a in arrays)
    pad = (hi - lo) * 0.05 or 1.0
    return (lo - pad, hi + pad)


# --- model (run のロードのみ。置換シミュ不要) -----------------------------------


def model_entries(
    bundles: dict[str, SurrogateBundle],
    net: NeuronGraph | None = None,
    comps: list[int] | None = None,
    i_ext_ylim: tuple[float, float] | None = None,
) -> list[SaveEntry]:
    """比べる N 本それぞれの自己記述図 (`<model>/...`) + run 横断の学習側サマリ表。

    **全 run 分描く** — レポートの単位が 1 系列 × N モデルなので N は比べたい本数
    そのもの (代表 1 本で済ませる必要がない)。学習データ図は `train_xr` の再生成を
    伴うが、それは N 本を比べると決めた分のコスト。
    """
    names = run_names(bundles)
    entries = [
        SaveEntry(name, df, sources=tuple(bundles))
        for name, df in summary_df(
            {names[run_id]: bundle for run_id, bundle in bundles.items()}
        )
    ]
    for run_id, bundle in bundles.items():
        entries += [
            SaveEntry(f"{slug(names[run_id])}/{name}", fig, sources=(run_id,))
            for name, fig in surrogate_figs(bundle, net, comps, i_ext_ylim)
        ]
    return entries


# --- eval (系列の結果: 格子 + 選択点の詳細図 + 点軸メトリクス) ------------------


def _detail_entries(
    view: SeriesView,
    bundles: dict[str, SurrogateBundle],
    names: dict[str, str],
    report: Report,
    tuning: Tuning,
) -> list[SaveEntry]:
    """選択した 1 点 × 各モデルの詳細図 + メトリクス表 (`<model>/p<点>/...`)。
    点 index を名前に入れるので、つまみを動かしても前の点を上書きしない。

    潜在射影は run ごとの surrogate が要るので bundles から引く (結果 artifact は
    surrogate を持たない = 描画側が run_id で対応付ける)。
    """
    index = view.clamp(tuning.detail_point)
    net = view.net
    comp_id = net.name_to_idx(report.eval_comp)
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
            _comp_ids(report.view_comps, net),
        )
        metrics = wave_report(
            dm_of(orig, surr, comp_id), tuning.spike_orig, tuning.spike_surr
        )
        # run 表示名は凡例向けに改行/`/` を含む → 名前に混ぜる分だけ slug 化
        run = slug(names[run_id])
        entries += [
            SaveEntry(
                f"{run}/p{index}/{fname}", artifact, sources=view.sources, draw=report
            )
            for fname, artifact in (*figs, *metrics)
        ]
    return entries


def eval_entries(
    view: SeriesView,
    bundles: dict[str, SurrogateBundle],
    report: Report,
    tuning: Tuning = NO_TUNING,
) -> list[SaveEntry]:
    """系列の結果: 入力電流プレビュー → 波形格子 (点 × モデル) → 選択点の詳細図 →
    点軸メトリクス折れ線。折れ線は**点が 2 つ以上のときだけ** (単発で 1 点の折れ線を
    出さない)。"""
    names = run_names(bundles)
    entries = [
        SaveEntry(
            "current",
            current_preview_fig(view.points[0].spec),
            sources=view.sources,
            draw=report,
        ),
        SaveEntry(
            "traces",
            trace_grid_fig(view, names, report.eval_comp),
            sources=view.sources,
            draw=report,
        ),
    ]
    if len(view.points) > 1:
        entries.append(
            SaveEntry(
                "metric",
                metric_fig(
                    view, names, report.eval_comp, report.metric, tuning.metric_ylim
                ),
                sources=view.sources,
                draw=report,
            )
        )
    return entries + _detail_entries(view, bundles, names, report, tuning)


# --- 入口 ----------------------------------------------------------------------


def render_report(
    view: SeriesView,
    bundles: dict[str, SurrogateBundle],
    report: Report,
    dest: Path,
    tuning: Tuning = NO_TUNING,
) -> list[Path]:
    """1 レポート (1 系列 × N モデル) を組み立てて `dest` へ保存する唯一の入口。
    呼び出し側 (`scripts/marimo.py` の描画ボタン) は artifact 読込 + surrogate
    ロード (mlflow 依存) だけを持ち、組立/保存はここに委譲する。成果物ごとの由来
    (`sources`/`draw`) は各 `SaveEntry` が持ち、`meta.json` へは `save_entries` が
    そのまま落とす。

    `eval_comp` が適用先に無い = **手元の結果に対する表示設定の誤り**なので、黙って
    何かを描かずエラー図 1 枚を返して気付けるようにする。
    """
    use_style()
    if report.eval_comp not in view.net.names:
        # matplotlib テキストとして描かれる (CJK グリフ非対応) → 英語で書く。
        msg = f"{view.name}: eval_comp {report.eval_comp!r} not in {view.target!r}"
        return save_entries([SaveEntry("error", error_fig(msg))], dest)
    # 学習側 (train_raw) は全軌道を覆うレンジ、評価側 (diff) は panels_diff の
    # 発表用既定に任せる (共有すると学習パルスの最大値で評価 step が潰れる)。
    entries = model_entries(
        bundles,
        view.net,
        _comp_ids(report.view_comps, view.net),
        _i_ext_ylim(bundles, view),
    ) + eval_entries(view, bundles, report, tuning)
    return save_entries(entries, dest)


def load_and_render_report(
    view: SeriesView,
    report: Report,
    dest: Path,
    load_surrogate_model: Callable[[str], SurrogateBundle],
    tuning: Tuning = NO_TUNING,
) -> list[Path]:
    """surrogate の解決から `render_report` までを一括した唯一の入口。呼び出し側
    (`scripts/marimo.py` の描画ボタン) は結果の読込 (mlflow 依存) だけを持てばよい。
    surrogate は結果に焼き込まれていない (run_id で対応付くだけ) ので、閉包項が
    要る図のために `load_surrogate_model` で引き直す。"""
    bundles = {run_id: load_surrogate_model(run_id) for run_id in view.run_ids}
    return render_report(view, bundles, report, dest, tuning)
