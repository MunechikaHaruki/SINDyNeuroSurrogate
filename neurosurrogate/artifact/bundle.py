"""各ドメインの単一成果物を、保存側が扱う成果物列へ編成する。

**編成はここで閉じる** — どの図を作るか (`*_artifacts`) だけでなく、それを
どの段へ書くか (`save_report`) まで。保存側に残るのは「どこへ書かせるか」
(一時 dir を渡して MLflow へ流す) だけ。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from ..core.coords import transform_gate
from ..core.network import NeuronGraph
from ..sim.artifacts import (
    metric_artifact,
    summary_artifact,
    traces_artifact,
)
from ..sim.result import SeriesResults
from ..surrogate.artifacts.model import (
    closure_artifact,
    neuron_graph_artifact,
    preprocessor_artifact,
)
from ..surrogate.artifacts.train import (
    train_manifold_artifact,
    train_preprocessed_artifact,
    train_raw_artifact,
    train_recon_artifact,
    train_v_coverage_artifact,
)
from ..surrogate.model import Surrogate
from ..surrogate.runs import SurrogateRuns
from ..waveform.artifacts import (
    attractor_artifact,
    current_preview_artifact,
    diff_artifact,
    metrics_artifact,
    metrics_scalar_artifact,
    simple_artifact,
)
from ..waveform.dynamics import DynamicMetrics
from .model import Artifact, Artifacts
from .plotting import use_style


def surrogate_artifacts(
    surrogate: Surrogate, view_comps: tuple[str, ...] = ()
) -> Artifacts:
    """学習 run 1 本が自己記述できる成果物をまとめる。"""
    net = surrogate.spec.dataset.net
    comps = [net.name_to_idx(comp) for comp in view_comps] or None
    return Artifacts(
        (
            *(
                artifact
                for artifact in (
                    closure_artifact(surrogate.closure),
                    preprocessor_artifact(surrogate.preprocessor),
                )
                if artifact is not None
            ),
            neuron_graph_artifact(
                net,
                {node.name for node in net.nodes if surrogate.spec.replaceable(node)},
            ),
            train_raw_artifact(surrogate, comps),
            train_preprocessed_artifact(surrogate, comps),
            train_recon_artifact(surrogate, comps),
            train_v_coverage_artifact(surrogate, comps),
            train_manifold_artifact(surrogate, comps),
        )
    )


def report_artifacts(
    view: SeriesResults,
    runs: SurrogateRuns,
    eval_comp: str,
    tuning: dict[str, Any],
) -> Artifacts:
    """run 横断のサマリ・波形格子・点軸メトリクスをまとめる。"""
    artifacts = [
        summary_artifact(runs),
        traces_artifact(view, runs, eval_comp),
    ]
    if len(view.original_waves) > 1:
        # y レンジは 3 つのつまみ (auto/下限/上限) で入り、図には 1 値で渡る。
        ylim = (
            None if tuning["yauto"] else (float(tuning["ymin"]), float(tuning["ymax"]))
        )
        artifacts.append(
            metric_artifact(view, runs, eval_comp, str(tuning["metric"]), ylim)
        )
    return Artifacts(tuple(artifacts))


def original_artifacts(view: SeriesResults) -> Artifacts:
    """原系の波形だけで決まる成果物をまとめる。"""
    # 描くのは**先頭点の電流** (掃引値を埋めた複製)。`series.spec` は掃引値の入って
    # いないカタログ既定なので、掃引系列では実際に回した波形と食い違う。
    return Artifacts((current_preview_artifact(view.series.points[0]),))


def _preprocessed_latent(
    surrogate: Surrogate, net: NeuronGraph, ds: xr.Dataset, comp_id: int
) -> xr.Dataset:
    """comp_id ノードの原系ゲートを surrogate の latent 空間へ射影した (V, latent...)
    xr (詳細図用)。置換対象外 (学習ドメイン外) は latent 比較不可。
    """
    comp = net.nodes[comp_id]
    if not surrogate.spec.replaceable(comp):
        raise ValueError(
            f"comp {comp.name!r} is outside the trained domain -> latent comparison "
            f"not possible (trained type {surrogate.spec.comp_type.name!r})"
        )
    return transform_gate(surrogate.preprocessor, ds, comp_id)


def detail_artifacts(
    view: SeriesResults,
    run_id: str,
    surrogate: Surrogate,
    eval_comp: str,
    view_comps: tuple[str, ...],
    tuning: dict[str, Any],
) -> Artifacts:
    """選択した 1 点・1 モデルの波形成果物をまとめる。"""
    net = view.series.spec.net
    comp_id = net.name_to_idx(eval_comp)
    original, surrogate_wave = view.pair(
        int(tuning["detail_point"]), view.column(run_id)
    )

    latent = _preprocessed_latent(surrogate, net, original, comp_id)
    dm = DynamicMetrics(original, surrogate_wave, comp_id, view.series.spec.dt)
    spike_orig = int(tuning["spike_orig"])
    spike_surr = int(tuning["spike_surr"])
    return Artifacts(
        (
            diff_artifact(original, latent, surrogate_wave, comp_id),
            simple_artifact(
                original, [net.name_to_idx(comp) for comp in view_comps] or None
            ),
            attractor_artifact(latent, surrogate_wave, comp_id),
            metrics_artifact(dm, spike_orig, spike_surr),
            metrics_scalar_artifact(dm, spike_orig, spike_surr),
        )
    )


def save_report(
    view: SeriesResults,
    runs: SurrogateRuns,
    tuning: dict[str, Any],
    root: Path,
) -> None:
    """1 レポート分の成果物を `root` 以下へ**段ごとに書く** = 描く側の唯一の入口。

    3 段: 直下が run 横断の産物、`models/<段名>/` が比べた 1 本ずつの自己記述図、
    `series/<段名>/` が波形 1 本で決まるもの (原系は `series/original/`)。`models/` と
    `series/` で同じ段名を使うので、1 本の run を 2 段から同じ綴りで辿れる。段は
    ここに書いた path がそのまま = 中間の構造を挟まない。

    学習run名は `SurrogateRuns` がpathの1区切りとして有効と保証するため、凡例・表・
    保存段のすべてでそのまま使う。

    **つまみ (`tuning`) の階層を解くのはここだけ**: UI が持つ形のまま受け取り、
    `common` は共有値へ、`report` / `detail` は対応する成果物集約関数へ渡す。
    **キーは全部必須で、既定値も検証もここには無い** — 既定値は `mo.ui.dictionary` が
    持つ唯一の場所で、欠けていれば `KeyError` がそのまま出る (握って別の値で描くより、
    どのキーが来ていないかがそのまま分かる方がよい)。記録 (`tuning.json`) は解く前の
    姿をそのまま添えるので、UI と保存の間に中間の型を挟まない。
    """
    if len(runs) != len(view.run_ids):
        raise ValueError(
            f"surrogate と結果の run 軸が不一致 ({len(runs)} != {len(view.run_ids)})"
        )
    common = dict(tuning["common"])
    eval_comp = str(common["eval_comp"])
    view_comps = tuple(common["view_comps"])
    use_style()
    # つまみも成果物 1 件 (`tuning.json`) = 図・表と同じ経路で書く。
    Artifact("tuning", tuning).save(root)
    report_artifacts(view, runs, eval_comp, dict(tuning["report"])).save(root)
    original_artifacts(view).save(root / "series/original")
    for run_id, (run_name, surrogate) in zip(view.run_ids, runs, strict=True):
        surrogate_artifacts(surrogate, view_comps).save(root / "models" / run_name)
        detail_artifacts(
            view,
            run_id,
            surrogate,
            eval_comp,
            view_comps,
            dict(tuning["detail"]),
        ).save(root / "series" / run_name)
