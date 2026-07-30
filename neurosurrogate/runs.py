"""run 軸 (surrogate) を評価条件に掛けて結果を束ねる層。marimo/mlflow 非依存。

`eval.py` が「1 シミュとは何か / 何を回すか / どう回すか」を持つのに対し、ここは
**どの surrogate と掛け合わせて何本回すか**: 条件 (label) × run の直積を作り
(`expand`)、並走シミュして `SimKey → SimResult` のフラットな dict にする。

**束ねる型を持たない**: 掃引点も run も `SimSpec` 自身のフィールドなので、軸を
組み替える処理は呼び出し側 (`metrics.select`) が dict を舐めるだけ。

**永続化は関心でない**: 結果の保存/読込は MLflow の評価 experiment が持ち
(`scripts/mlflow_io.py`)、ここは「spec → 結果」だけを知る。`SimResult.source` は
その出所を指す不透明な識別子で、何を指すか (MLflow run) はここでは決めない。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import replace as dc_replace

import xarray as xr

from .eval import SimSpec, simulate
from .surrogate.bundle import SurrogateBundle
from .surrogate.meta import SurrogateMeta
from .surrogate.replace import replaced_names

# --- 条件 × run → 結果 --------------------------------------------------------------

SimKey = tuple[str, str | None]  # (label, run_id)。run_id=None は原系


@dataclass(frozen=True)
class SimResult:
    """1 SimSpec 分の実行結果 = 仕様 + 表示名 + 波形。"""

    spec: SimSpec
    run_label: str | None  # 表示名 (凡例/行見出し)。None=原系
    dataset: xr.Dataset
    source: str | None = None  # 出所の識別子 (実行直後は無い = None)


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


def run_results(
    specs: dict[str, SimSpec],
    surrogates: dict[str, SurrogateBundle],
    labels_by_run: dict[str, str],
) -> dict[SimKey, SimResult]:
    """`expand` した各 key を並走シミュし `SimKey → SimResult` を返す
    (**spec → 結果の唯一の入口**)。`labels_by_run` = run_id → 表示名。"""
    return {
        key: SimResult(
            spec,
            labels_by_run[key[1]] if key[1] is not None else None,
            simulate(spec, surrogates[key[1]] if key[1] is not None else None),
        )
        for key, spec in expand(specs, surrogates).items()
    }


# --- run 軸 (surrogate) の表示名と絞り込み --------------------------------------------


def usable(meta: SurrogateMeta, specs: Iterable[SimSpec]) -> bool:
    """宣言した評価の 1 本でも置換できる surrogate か = UI の run 絞り込み条件。"""
    return any(bool(replaced_names(meta, s.net)) for s in specs)


def dedupe_labels(names: list[str]) -> list[str]:
    """衝突した名前にだけ順序の連番を付ける (与えた順)。結果 dict のキーが silent に
    潰れて表と図が食い違うのを防ぐ共通規約 (選択を拒否せず全部見せる)。"""
    counts = Counter(names)
    seen: Counter[str] = Counter()
    labels = []
    for name in names:
        seen[name] += 1
        labels.append(name if counts[name] == 1 else f"{name}#{seen[name]}")
    return labels


def run_labels(surrogates: list[SurrogateBundle]) -> list[str]:
    """surrogate 列の表示名 (与えた順)。

    `meta.label` は学習構造 + 学習データまでしか区別しない → library_specs 違いや
    同 config の再実行は同じ label になるため連番で潰れを防ぐ。
    """
    return dedupe_labels([s.meta.label for s in surrogates])
