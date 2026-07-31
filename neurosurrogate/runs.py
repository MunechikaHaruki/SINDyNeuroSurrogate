"""run 軸 (surrogate) を評価条件に掛けて結果を束ねる層。marimo/mlflow 非依存。

`eval.py` が「1 シミュとは何か / 何を回すか / どう回すか」を持つのに対し、ここは
**どの surrogate と掛け合わせて何本回すか**: 系列 (点列) × run の直積を作り
(`expand`)、並走シミュして `SimKey → SimResult` のフラットな dict にする。

**label と系列名はここの関心**: `SimSpec` は純粋な計算入力で識別を持たないので、
`EVALS` のキー (系列名) から label (単発は `系列名`、掃引は `系列名#i`) を作るのは
`expand` 1 箇所。結果側は `SimResult.series` として持ち回る。

**束ねる型を持たない**: 軸を組み替える処理は呼び出し側 (`metrics.select`) が dict を
舐めるだけ。

**永続化は関心でない**: 結果の保存/読込は MLflow の評価 experiment が持ち
(`scripts/mlflow_io.py`)、ここは「spec → 結果」だけを知る。`SimResult.source` は
その出所を指す不透明な識別子で、何を指すか (MLflow run) はここでは決めない。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import xarray as xr

from .eval import EvalSeries, SimSpec, simulate
from .surrogate.bundle import SurrogateBundle
from .surrogate.meta import SurrogateMeta
from .surrogate.replace import replaced_names

# --- 条件 × run → 結果 --------------------------------------------------------------

SimKey = tuple[str, str | None]  # (label, run_id)。run_id=None は原系


@dataclass(frozen=True)
class SimResult:
    """1 シミュの結果 = 入力 (`SimSpec`) + それがどこの何だったか (識別) + 波形。

    **識別はすべてこちら側**: 系列名・軸・どの surrogate で回したか (`run_id`)・
    表示名・出所。`SimSpec` は純粋な計算入力に保つ。"""

    spec: SimSpec
    series: str  # 系列名 (`EVALS` のキー。掃引しても不変 = 図の系列識別)
    axis: str | None  # 掃引軸の電流パラメータ名 (None=単発)。図の x 軸
    run_id: str | None  # どの surrogate で回したか (None=原系)
    run_label: str | None  # 表示名 (凡例/行見出し)。None=原系
    dataset: xr.Dataset
    source: str | None = None  # 出所の識別子 (実行直後は無い = None)

    @property
    def point(self) -> float | None:
        """軸上の位置 (単発なら None)。`current_params` に確定済みの値を読むだけ
        — 二重に持たない。"""
        return float(self.spec.current_params[self.axis]) if self.axis else None


def labels(evals: dict[str, EvalSeries]) -> dict[str, tuple[str, SimSpec]]:
    """系列名 → `EvalSeries` を label → (系列名, 点) へ平らにする。**label 規約は
    ここだけ**: 単発は系列名そのもの、掃引は `系列名#i` (点の並び順)。"""
    out: dict[str, tuple[str, SimSpec]] = {}
    for series, ev in evals.items():
        if len(ev.points) == 1:
            out[series] = (series, ev.points[0])
        else:
            out.update({f"{series}#{i}": (series, p) for i, p in enumerate(ev.points)})
    return out


def expand(
    evals: dict[str, EvalSeries], surrogates: dict[str, SurrogateBundle]
) -> dict[SimKey, tuple[str, SimSpec]]:
    """label × (原系 + 置換可能な run) の直積 → `SimKey → (系列名, spec)`。原系は
    `run_id=None` として同じ経路 (特別扱いしない)。1 本も置換できない label は落とす。
    **spec は run に依らず同一** — どの surrogate で回すかは key 側 (`run_id`) が
    持ち、`simulate` には引数で渡る。
    """
    out: dict[SimKey, tuple[str, SimSpec]] = {}
    for label, (series, spec) in labels(evals).items():
        compatible = [
            run_id
            for run_id, s in surrogates.items()
            if replaced_names(s.meta, spec.net)
        ]
        if not compatible:
            continue
        out[(label, None)] = (series, spec)
        for run_id in compatible:
            out[(label, run_id)] = (series, spec)
    return out


def run_results(
    evals: dict[str, EvalSeries],
    surrogates: dict[str, SurrogateBundle],
    labels_by_run: dict[str, str],
) -> dict[SimKey, SimResult]:
    """`expand` した各 key を並走シミュし `SimKey → SimResult` を返す
    (**条件 → 結果の唯一の入口**)。`labels_by_run` = run_id → 表示名。"""
    return {
        (label, run_id): SimResult(
            spec,
            series,
            evals[series].axis,
            run_id,
            labels_by_run[run_id] if run_id is not None else None,
            simulate(spec, surrogates[run_id] if run_id is not None else None),
        )
        for (label, run_id), (series, spec) in expand(evals, surrogates).items()
    }


# --- run 軸 (surrogate) の表示名と絞り込み --------------------------------------------


def usable(meta: SurrogateMeta, evals: dict[str, EvalSeries]) -> bool:
    """宣言した評価の 1 本でも置換できる surrogate か = UI の run 絞り込み条件。"""
    return any(
        bool(replaced_names(meta, spec.net))
        for ev in evals.values()
        for spec in ev.points
    )


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
