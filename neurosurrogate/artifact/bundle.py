"""各ドメインの成果物を集めて、**どの段へ書くか**を決める唯一の場所。

**何を描くかはここに無い** — 成果物の集合 (`Artifacts`) はそれを持つドメインが
自分で組む (`sim.artifacts` = run 横断・原系・1 ペア、`surrogate.artifacts` =
学習 run 1 本)。
ここが持つのは、ドメインを跨ぐ組み立て (原系ゲートの latent 射影) と、つまみを
解いて段へ配ること (`save_report`) だけ。保存側に残るのは「どこへ書かせるか」
(一時 dir を渡して MLflow へ流す) だけ。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from ..core.coords import transform_gate
from ..core.network import NeuronGraph
from ..sim.artifacts import (
    detail_artifacts,
    original_artifacts,
    report_artifacts,
)
from ..sim.result import SeriesResults
from ..surrogate.artifacts import surrogate_artifacts
from ..surrogate.model import Surrogate
from ..surrogate.runs import SurrogateRuns
from .model import Artifact
from .plotting import use_style


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
    net = view.series.spec.net
    comp_id = net.name_to_idx(eval_comp)
    comps = [net.name_to_idx(comp) for comp in view_comps] or None
    report, detail = dict(tuning["report"]), dict(tuning["detail"])
    use_style()
    # つまみも成果物 1 件 (`tuning.json`) = 図・表と同じ経路で書く。
    Artifact("tuning", tuning).save(root)
    report_artifacts(
        view,
        runs,
        eval_comp,
        str(report["metric"]),
        # y レンジは 3 つのつまみ (auto/下限/上限) で入り、図には 1 値で渡る。
        None if report["yauto"] else (float(report["ymin"]), float(report["ymax"])),
    ).save(root)
    original_artifacts(view).save(root / "series/original")
    for run_id, (run_name, surrogate) in zip(view.run_ids, runs, strict=True):
        surrogate_artifacts(surrogate, view_comps).save(root / "models" / run_name)
        original, surrogate_wave = view.pair(
            int(detail["detail_point"]), view.column(run_id)
        )
        detail_artifacts(
            original,
            _preprocessed_latent(surrogate, net, original, comp_id),
            surrogate_wave,
            comp_id,
            view.series.spec.dt,
            comps,
            int(detail["spike_orig"]),
            int(detail["spike_surr"]),
        ).save(root / "series" / run_name)
