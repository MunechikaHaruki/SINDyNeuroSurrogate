"""**実行**: 記述 (`spec`) と置換器を受け取って結果 (`result`) を作る段。

記述は置換器を知らない (`EvalSeries` は「何を回すか」だけ) → **仕様 × surrogate を
掛け合わせるのはここだけ**。run 軸を掛ける `replaced_runs` も同じ理由でここに居る。

返りは列 (`SeriesRun`) = 保存もキャッシュもこの単位。
"""

from __future__ import annotations

import xarray as xr

from ..core.diverge import log_divergence
from ..core.simulator import unified_simulator
from ..surrogate.bundle import SurrogateBundle, SurrogateRuns
from ..surrogate.meta import SurrogateMeta
from ..surrogate.replace import apply_surrogate
from ..surrogate.replace import replaceable as node_replaceable
from .result import SeriesRun
from .spec import EvalSeries, SimSpec


def simulate(spec: SimSpec, surrogate: SurrogateBundle | None) -> xr.Dataset:
    """1 シミュ → 波形。`surrogate=None` なら原系、あれば `apply_surrogate` してから
    回す。**入力は返さない** (呼んだ側が既に持っている)。"""
    dset = spec.materialize()
    if surrogate is None:
        return unified_simulator(dset)
    surr_ds = unified_simulator(apply_surrogate(surrogate, dset))
    # 系列名は spec が持たない (カタログのキーが単一源) → 入力そのもので名乗る。
    where = f"{spec.target}/{spec.current_type} / {surrogate.meta.surr_type_name}"
    log_divergence(spec.net, surr_ds, where)
    return surr_ds


def run_column(
    series: EvalSeries,
    run_id: str | None,
    surrogate: SurrogateBundle | None,
) -> SeriesRun:
    """掃引の点列を順に回して**1 列**にする (**系列 → 結果の唯一の入口**)。

    `run_id` は回した置換器の出所 (学習 run の id) で、両方 `None` なら原系。波形は
    `series.points` と同じ並びで、点ごとの仕様も掃引軸も添えない (どちらも記述の側に
    ある)。"""
    if (run_id is None) != (surrogate is None):
        raise ValueError(f"run_id と置換器が対でない (run {run_id})")
    waves = [simulate(spec, surrogate) for spec in series.points]
    return SeriesRun(series, run_id, waves)


def replaceable(series: EvalSeries, meta: SurrogateMeta) -> bool:
    """この系列を置換できる surrogate か = 適用先に置換されるノードが 1 つでも
    あるか (点は適用先を変えないので `spec` で判定)。"""
    return any(node_replaceable(meta, n) for n in series.spec.net.nodes)


def replaced_runs(series: EvalSeries, runs: SurrogateRuns) -> SurrogateRuns:
    """1 系列 × 学習 run → 実際にこの系列を置換できる run だけ。**run 軸を絞る唯一の
    場所**。

    置換できない run は落ちる (回しても比較にならない)。空になった選択を飛ばすか
    拒むかは呼び出し側の関心 (その場で回す側は飛ばし、保存側は拒む)。
    """
    return SurrogateRuns(
        tuple(
            (run_name, bundle)
            for run_name, bundle in runs
            if replaceable(series, bundle.meta)
        )
    )
