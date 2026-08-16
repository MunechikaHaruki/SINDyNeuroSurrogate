"""**波形 1 本について描ける図**: 原系の入力電流と、選択した 1 点 × **1 モデル**の
詳細図。marimo/MLflow 非依存。

run 横断でない = 同じ波形を別のレポートで見ても同じ図 (だから保存段も評価 run 側で、
レポートを増やしても複製されない)。run 横断の図は隣の `report` module、置換シミュの
結果を受け取らない図 (学習 run 1 本の自己記述) は `surrogate.figures`。

**単発と掃引で経路を分けない** — 点が 1 つでも点 index を名前に持つ 1 組が出るだけ。
"""

from __future__ import annotations

from dataclasses import replace

from ...plotting import Artifact, use_style
from ...surrogate.bundle import SurrogateBundle
from ...surrogate.diagnostics import preprocessed_latent
from ...waveform import cell_figs, current_preview_fig, dm_of, wave_report
from ..eval import SeriesResults


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
