"""サロゲート評価: spec を原系/置換系で並走シミュする。marimo/mlflow 非依存の純粋
ドメイン層 (widget は scripts 側)。

**結果は 1 シミュ = 1 key = 1 Dataset のフラットな dict**: `SimKey = (label, run_id)`。
掃引点も run もどちらも `SimSpec` 自身のフィールドなので、束ねる型 (旧 `EvalGrid`) を
持たない — 軸を組み替える処理は呼び出し側 (`metrics.select`) が dict を舐めるだけ。

**永続化はここの関心でない**: 結果の保存/読込は MLflow の評価 experiment が持ち
(`scripts/mlflow_io.py`)、この層は「spec → 結果」だけを知る。`SimResult.source` は
その出所を指す不透明な識別子で、何を指すか (MLflow run) はここでは決めない。
"""

from dataclasses import dataclass
from dataclasses import replace as dc_replace

import xarray as xr

from ..core.diverge import log_divergence
from ..core.simulator import unified_simulator
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.replace import apply_surrogate, replaced_names
from .spec import SimSpec

SimKey = tuple[str, str | None]  # (label, run_id)。run_id=None は原系


@dataclass(frozen=True)
class SimResult:
    """1 SimSpec 分の実行結果 = 仕様 + 表示名 + 波形。"""

    spec: SimSpec
    run_label: str | None  # 表示名 (凡例/行見出し)。None=原系
    dataset: xr.Dataset
    source: str | None = None  # 出所の識別子 (実行直後は無い = None)


# --- 実行 (spec → シミュ) -----------------------------------------------------------


def expand(
    specs: dict[str, SimSpec], surrogates: dict[str, SurrogateBundle]
) -> dict[SimKey, SimSpec]:
    """label × (原系 + 置換可能な run) の直積。原系は `run_id=None` として同じ経路
    (特別扱いしない)。1 本も置換できない label は落とす。返す spec は `run_id`
    フィールドを key と一致させて埋める (呼び出し側が key から surrogate を引ける)。
    """
    out: dict[SimKey, SimSpec] = {}
    for label, spec in specs.items():
        compatible = {
            run_id: s
            for run_id, s in surrogates.items()
            if replaced_names(s.meta, spec.net)
        }
        if not compatible:
            continue
        out[(label, None)] = spec
        for run_id in compatible:
            out[(label, run_id)] = dc_replace(spec, run_id=run_id)
    return out


def simulate(spec: SimSpec, surrogate: SurrogateBundle | None) -> xr.Dataset:
    """1 シミュ。`surrogate=None` なら原系、あれば `apply_surrogate` してから回す。"""
    dset = spec.dataset()
    if surrogate is None:
        return unified_simulator(dset)
    surr_ds = unified_simulator(apply_surrogate(surrogate, dset))
    log_divergence(spec.net, surr_ds, f"{spec.name} / {surrogate.meta.label}")
    return surr_ds


def run_results(
    specs: dict[str, SimSpec],
    surrogates: dict[str, SurrogateBundle],
    run_labels: dict[str, str],
) -> dict[SimKey, SimResult]:
    """`expand` した各 key を並走シミュし `SimKey → SimResult` を返す
    (**spec → 結果の唯一の入口**)。`run_labels` = run_id → 表示名。"""
    return {
        key: SimResult(
            spec,
            run_labels[key[1]] if key[1] is not None else None,
            simulate(spec, surrogates[key[1]] if key[1] is not None else None),
        )
        for key, spec in expand(specs, surrogates).items()
    }
