"""**実行**: 記述 (`spec`) と置換器を受け取って結果 (`result`) を作る段。

記述は置換器を知らない (`EvalSeries` は「何を回すか」だけ) → **仕様 × surrogate を
掛け合わせるのはここだけ**。どの run が置換できるかの絞り込みは回す前に決まるので
持たない (`SurrogateRuns.replacing`)。

返りは列 (`SeriesRun`) = 保存もキャッシュもこの単位。
"""

from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from ..core.diverge import log_divergence
from ..core.simulator import unified_simulator
from ..surrogate.model import Surrogate
from .result import SeriesRun
from .spec import EvalSeries, SimSpec


def simulate(
    spec: SimSpec, surrogate: Surrogate | None, targets: Sequence[str]
) -> xr.Dataset:
    """1 シミュ → 波形。`surrogate=None` なら原系 (`targets` は使わない)、あれば
    `targets` のノードを置換してから回す。**入力は返さない** (呼んだ側が既に持つ)。
    """
    dset = spec.materialize()
    if surrogate is None:
        return unified_simulator(dset)
    surr_ds = unified_simulator(surrogate.apply(dset, targets))
    # 系列名は spec が持たない (カタログのキーが単一源) → 入力そのもので名乗る。
    where = f"{spec.target}/{spec.current_type} / {surrogate.spec.surr_type_name()}"
    log_divergence(spec.net, surr_ds, where)
    return surr_ds


def run_column(
    series: EvalSeries,
    run_id: str | None,
    surrogate: Surrogate | None,
) -> SeriesRun:
    """掃引の点列を順に回して**1 列**にする (**系列 → 結果の唯一の入口**)。

    `run_id` は回した置換器の出所 (学習 run の id) で、両方 `None` なら原系。波形は
    `series.points` と同じ並びで、点ごとの仕様も掃引軸も添えない (どちらも記述の側に
    ある)。"""
    if (run_id is None) != (surrogate is None):
        raise ValueError(f"run_id と置換器が対でない (run {run_id})")
    waves = [
        simulate(spec, surrogate, series.replace_targets) for spec in series.points
    ]
    return SeriesRun(series, run_id, waves)
