"""**掃引の結果**: 1 列 (`SeriesRun`) と、原系 + 置換系を run 軸に並べた束
(`SeriesResults`)。

**列が単位**: 1 列 = 1 系列を 1 つの置換器 (原系なら無し) で回した波形の並び。
決定的シミュを「同じ入力なら回さない」でキャッシュする単位も、永続化の単位
(`mlflow_io.series` の 1 run) も同じこれ — **ドメインと保存で単位が一致する**。

**結果は波形そのもの** — 1 点の計算入力は記述 (`EvalSeries.points`) が持つので、
波形を包む型は要らない (点の位置 = list の添字が両者を対応させる)。

**計算も描画もしない** — 回すのは `run`、図に落とすのは `artifacts`。結果を扱う層
(指標/描画) が見るのはここだけ = 「点を値で並べ直す」「run 軸を数える」を各図が
再実装しない。
"""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from ..core.network import NeuronGraph
from .spec import EvalSeries


@dataclass(frozen=True)
class SeriesRun:
    """**1 列** = 記述 (`series`) を 1 つの置換器で回した波形の並び。

    `run_id` は**どの置換器で回したか**の標識で、`None` が原系。学習 run の id を
    そのまま使う (置換器そのものは持たない — 結果は置換器を知らなくてよく、表示名も
    `surrogate` 側から解く)。

    `waves[i]` の計算入力は `series.points[i]`。点ごとの仕様も識別子も持たず、対応は
    添字だけ = 保存も「記述 1 本 + 波形の列」で済む。
    """

    series: EvalSeries
    run_id: str | None  # 学習 run の id (None = 原系)
    waves: list[xr.Dataset]  # `series.points` と同じ並び

    def __post_init__(self) -> None:
        if len(self.waves) != len(self.series.points):
            raise ValueError(
                f"点数が記述の {len(self.series.points)} 点と揃わない "
                f"({len(self.waves)} 本, run {self.run_id})"
            )


@dataclass(frozen=True)
class SeriesResults:
    """**1 系列を run 軸に開いた束**: 原系 1 列 (`original`) と置換系の列
    (`surrs`、与えた順 = 凡例/行見出しの並び)。全列が同じ記述を回したものであることを
    構築時に保証する。

    **run_id は列が持つ**ので、束は id をキーに持たない (同じ id が 2 箇所に載らない)。
    軸まわり (`net` / `target` / `axis` / `values` / `dt`) は記述を読むだけで、波形から
    復元しない。

    **由来は持たない**: 系列名も、どの評価 run から読んだかという MLflow の同一性も
    無い (解くのは `mlflow_io.report`)。
    """

    original: SeriesRun
    surrs: tuple[SeriesRun, ...]

    def __post_init__(self) -> None:
        # 束ねてよいのは**同じ掃引を回した列だけ** (点軸が揃わない列を図の側で
        # 検出させない)。列ごとの点数は `SeriesRun` が既に見ている。
        key = self.series.hash()
        if any(column.series.hash() != key for column in self.surrs):
            raise ValueError("記述の違う列は 1 つの束にできない")
        if self.original.run_id is not None:
            raise ValueError(f"原系の列に run_id {self.original.run_id}")
        ids = [column.run_id for column in self.surrs]
        if None in ids or len(set(ids)) != len(ids):
            raise ValueError(f"置換系の run_id が欠けるか重複 {ids}")

    @property
    def series(self) -> EvalSeries:
        """回した掃引の記述 (全列で同じ)。"""
        return self.original.series

    @property
    def points(self) -> list[xr.Dataset]:
        """原系の波形 (点の値順)。点軸の長さもこれが持つ。"""
        return self.original.waves

    @property
    def run_ids(self) -> list[str]:
        """置換系の run_id (列の順。原系は含まない)。"""
        return [str(column.run_id) for column in self.surrs]

    @property
    def net(self) -> NeuronGraph:
        return self.series.spec.net

    @property
    def target(self) -> str:
        return self.series.spec.target

    @property
    def axis(self) -> str | None:
        """掃引した電流パラメータ名 (単発なら None)。図の x 軸。"""
        return self.series.param

    @property
    def values(self) -> list[float | None]:
        """点軸の値 (単発なら `[None]`)。列見出しと折れ線の x に使う。"""
        return self.series.axis_values

    @property
    def dt(self) -> float:
        """刻み幅 (点は変えない)。波形から時間軸を測る側が引く。"""
        return self.series.spec.dt

    def column(self, run_id: str) -> SeriesRun:
        """学習 run の id → その置換系の列。"""
        for column in self.surrs:
            if column.run_id == run_id:
                return column
        raise KeyError(f"run {run_id} の列が束に無い")

    def pair(self, index: int, column: SeriesRun) -> tuple[xr.Dataset, xr.Dataset]:
        """点 `index` の (原系, 置換系) の波形。"""
        return self.points[index], column.waves[index]
