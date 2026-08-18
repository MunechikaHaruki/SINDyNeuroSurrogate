"""**実験の記述**: 1 シミュの計算入力 (`SimSpec`)、1 回の掃引 (`EvalSeries`)。
それぞれ `result` の 波形 / `SeriesRun` と 1 対 1。

**実行を知らない** = ここに書けるのは「何を回すか」だけで、どの surrogate で回すかも
回した結果も持たない (実行は `run`、結果は `result`)。おかげで**同一性が記述だけで
決まり** (`hash` を持つのは再利用の単位である `EvalSeries`)、
surrogate 層にも依存しない (`surrogate.meta` がこの module を import する)。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Self

import numpy as np

from ..core.network import DatasetConfig, NeuronGraph
from .catalog.currents import CURRENT_MAP
from .catalog.targets import MCMODELS


@dataclass(frozen=True, kw_only=True)
class SimSpec:
    """1 回のシミュレーションの仕様 = **純粋な計算入力**: 適用先 target × 電流
    (掃引点は `current_params` に確定済み)。これだけで波形が決まる。

    **識別は一切持たない** — 系列名はカタログのキー、どの surrogate で回すかは
    `run.simulate` の引数、掃引の中の位置は `EvalSeries` 側。同一性 (`hash`) も
    持たない: 「既に回したか」を引くのは保存の単位 (`EvalSeries`) で、この型は
    `to_dict` でその中身として運ばれる。
    """

    target: str
    current_type: str
    dt: float
    current_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """永続化 (MLflow の学習 run / 評価 run) が仕様として持ち回る形へ。
        **名前だけを書く** = 適用先のグラフも電流の波形も保存しない (どちらも
        `MCMODELS` / `CURRENT_MAP` から名前で解ける導出物)。"""
        return {
            "target": self.target,
            "current_type": self.current_type,
            "dt": self.dt,
            "current_params": self.current_params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            target=str(d["target"]),
            current_type=str(d["current_type"]),
            dt=float(d["dt"]),
            current_params=dict(d.get("current_params") or {}),
        )

    @property
    def net(self) -> NeuronGraph:
        """適用先の MC モデル。**target 名 → ネットの解決はここだけ**が行い、結果型も
        描画も spec 経由で引く (文字列キーを持ち回らない)。"""
        return MCMODELS[self.target]

    def current(self) -> np.ndarray:
        """電流波形。名前 → 波形の解決も名前を持つこの型の関心。"""
        return CURRENT_MAP[self.current_type](**self.current_params)(self.dt)

    def materialize(self) -> DatasetConfig:
        """仕様 → 実行入力 (原系)。置換系は `surrogate.replace.apply_surrogate` が
        この実行入力から非破壊で作る。"""
        return DatasetConfig(dt=self.dt, net=self.net, current=self.current())


@dataclass(frozen=True)
class EvalSeries:
    """**1 回の掃引実験の記述**: 何を (`spec`)・どの電流パラメータで振るか
    (`param`/`values`)。`param` を渡さなければ単発 (点 1 つ) で、以降は掃引と同じ
    経路を通る。

    **どの surrogate で回すかも run 軸も持たない**: 置換器を掛けるのは `run`、
    run 軸に開いた結果を持つのは `result.SeriesResults` で、どちらもこの型の外に居る。
    掃引の記述が置換器から独立している = 原系の再利用が hash 1 本で効く。

    **保存の単位でもある**: 1 系列 = 1 評価 run (点列を丸ごと 1 artifact に持つ)
    なので、「同じ掃引を既に回したか」を引く鍵 (`hash`) と往復の形 (`to_dict` /
    `from_dict`) をこの型が持つ。
    """

    spec: SimSpec
    param: str | None = None  # 掃引する電流パラメータ名 (None=単発)。図の x 軸
    values: Sequence[float] = ()  # 掃引点の値列 (等間隔でなくてもよい)

    def to_dict(self) -> dict:
        """永続化 (評価 run の param) が持ち回る形 = 掃引の定義そのもの。"""
        return {
            "spec": self.spec.to_dict(),
            "param": self.param,
            "values": [float(v) for v in self.values],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            spec=SimSpec.from_dict(d["spec"]),
            param=d["param"] or None,
            values=[float(v) for v in d["values"]],
        )

    def hash(self) -> str:
        """**同じ掃引を既に回したか**の鍵 (置換器は記述に含まれない = 原系の再利用が
        これ 1 本で効く)。置換系は呼び出し側がここに run_id を組む。

        完全な仕様は波形 run 側が別に持つので、短縮ハッシュで足りる。"""
        key = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha1(key.encode()).hexdigest()[:8]

    @property
    def points(self) -> list[SimSpec]:
        """点ごとの計算入力 (値順)。`current_params[param]` に値を埋めた複製 =
        `spec` と `values` からの派生で、二重に持たない。"""
        if self.param is None:
            return [self.spec]
        return [
            dc_replace(
                self.spec,
                current_params={**self.spec.current_params, self.param: float(v)},
            )
            for v in self.values
        ]

    @property
    def axis_values(self) -> list[float | None]:
        """`points` と対になる点軸の値 (単発なら `[None]`)。図の x 値・列見出し。
        単発を「値の無い点 1 つ」に退化させる場所を `points` と揃えて 1 箇所に置く。"""
        if self.param is None:
            return [None]
        return [float(v) for v in self.values]
