"""**シミュ結果の器**: 1 シミュ (`SimResult`) と、1 系列を点軸 × run 軸に開いた並び
(`SeriesResults`)。

**計算も描画もしない** — 回すのは `run`、図に落とすのは `figures`。結果を扱う層
(指標/描画) が見るのはここだけ = 「点を値で並べ直す」「run 軸を数える」を各図が
再実装しない。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import xarray as xr

from ..core.network import NeuronGraph
from .spec import EvalSeries, SimSpec


@dataclass(frozen=True)
class SimResult:
    """1 シミュの結果 = 入力 (`SimSpec`) + 波形。`run.simulate` の返り値。

    **どこの何だったか (系列名 / 点 index / どの run / どこに保存されたか) は
    持たない**: 系列の中の位置は `run.run_points` が返す並び順、run はそれを
    呼んだ側が知っている。結果を集めて軸を張るのは `SeriesResults`、保存先の id は
    永続化層の関心。"""

    spec: SimSpec
    dataset: xr.Dataset
    # 系列の中で振られていた電流パラメータ名 (単発 / 系列の外で回した = None)。1 シミュ
    # には無い情報なので `run.run_points` が書き足す欄で、図の x 軸に使う。
    axis: str | None = None

    @property
    def point(self) -> float | None:
        """軸上の位置 (単発なら None)。`current_params` に確定済みの値を読むだけ
        — 二重に持たない。"""
        return float(self.spec.current_params[self.axis]) if self.axis else None


def attach(series: EvalSeries, datasets: Sequence[xr.Dataset]) -> list[SimResult]:
    """保存済みの波形列 → 点列の `SimResult` (**再シミュ無しの `run.run_points`**)。

    点の並びと各点の計算入力は `EvalSeries.points` が単一源 = 波形さえ順に保存して
    あれば点ごとの識別子を持ち回らずに復元できる。"""
    return [
        SimResult(spec, ds, axis=series.param)
        for spec, ds in zip(series.points, datasets, strict=True)
    ]


@dataclass(frozen=True)
class SeriesResults:
    """**1 系列を両軸に開いた結果**: 点軸に値順で並んだ原系 (`points`) と、run ごとに
    同じ並びの置換系 (`surrs`)。列の長さは常に揃う。

    **持つのは結果だけ**の素のデータ: 系列名も、どの評価 run から読んだかという
    MLflow の同一性も持たない (由来と保存段を解くのは `mlflow_io.report.Report`)。
    点の位置は list の添字そのもので、平坦キーは出てこない。

    軸の値 (`net` / `target` / `axis` / `values`) はどれも点が既に持つものを読むだけ
    = 二重に持たない (点は適用先も掃引軸も変えないので先頭から引ける)。
    """

    points: list[SimResult]  # 原系 (掃引点の値順。単発なら 1 件)
    surrs: dict[str, list[SimResult]]  # run_id → `points` と同じ並びの置換系

    def __post_init__(self) -> None:
        bad = [rid for rid, col in self.surrs.items() if len(col) != len(self.points)]
        if bad:
            raise ValueError(f"点数が原系と揃わない run {bad}")

    @property
    def run_ids(self) -> list[str]:
        """置換系の run_id (与えた順。原系は含まない)。"""
        return list(self.surrs)

    @property
    def net(self) -> NeuronGraph:
        return self.points[0].spec.net

    @property
    def target(self) -> str:
        return self.points[0].spec.target

    @property
    def axis(self) -> str | None:
        """掃引した電流パラメータ名 (単発なら None)。図の x 軸。"""
        return self.points[0].axis

    @property
    def values(self) -> list[float | None]:
        """点軸の値 (単発なら `[None]`)。列見出しと折れ線の x に使う。"""
        return [r.point for r in self.points]

    def pair(self, index: int, run_id: str) -> tuple[SimResult, SimResult]:
        """点 `index` の (原系, 置換系)。"""
        return self.points[index], self.surrs[run_id][index]
