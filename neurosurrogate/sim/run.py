"""**実行**: 記述 (`spec`) と置換器を受け取って結果 (`result`) を作る段。

記述も結果も置換器を知らない (`EvalSeries` は「何を回すか」だけ、`SimResult` は
波形だけ) → **仕様 × surrogate を掛け合わせるのはここだけ**。run 軸を掛ける
`replaced_runs` も同じ理由でここに居る。
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from ..core.diverge import log_divergence
from ..core.simulator import unified_simulator
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.meta import SurrogateMeta
from ..surrogate.replace import apply_surrogate
from ..surrogate.replace import replaceable as node_replaceable
from .result import SimResult
from .spec import EvalSeries, SimSpec


def simulate(spec: SimSpec, surrogate: SurrogateBundle | None) -> SimResult:
    """1 シミュ。`surrogate=None` なら原系、あれば `apply_surrogate` してから回す。"""
    dset = spec.materialize()
    if surrogate is None:
        return SimResult(spec, unified_simulator(dset))
    surr_ds = unified_simulator(apply_surrogate(surrogate, dset))
    # 系列名は spec が持たない (カタログのキーが単一源) → 入力そのもので名乗る。
    where = f"{spec.target}/{spec.current_type} / {surrogate.meta.label}"
    log_divergence(spec.net, surr_ds, where)
    return SimResult(spec, surr_ds)


def run_points(
    series: EvalSeries, surrogate: SurrogateBundle | None
) -> list[SimResult]:
    """掃引の点列を順に回す (**系列 → 結果の唯一の入口**)。返りは点の並び順で、素の
    結果に掃引軸だけ書き足す (1 シミュは自分が何の軸の上に居るかを知らない)。"""
    return [
        dc_replace(simulate(spec, surrogate), axis=series.param)
        for spec in series.points
    ]


def replaceable(series: EvalSeries, meta: SurrogateMeta) -> bool:
    """この系列を置換できる surrogate か = 適用先に置換されるノードが 1 つでも
    あるか (点は適用先を変えないので `spec` で判定)。"""
    return any(node_replaceable(meta, n) for n in series.spec.net.nodes)


def replaced_runs(
    series: EvalSeries, bundles: dict[str, SurrogateBundle]
) -> dict[str, SurrogateBundle]:
    """1 系列 × 学習 run → 実際にこの系列を置換できる run だけ。**run 軸を絞る唯一の
    場所**。

    置換できない run は落ちる (回しても比較にならない)。空になった選択を飛ばすか
    拒むかは呼び出し側の関心 (その場で回す側は飛ばし、保存側は拒む)。
    """
    return {
        run_id: bundle
        for run_id, bundle in bundles.items()
        if replaceable(series, bundle.meta)
    }
