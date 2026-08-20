"""**実験の記述**: 1 シミュの計算入力 (`SimSpec`)、1 回の掃引 (`EvalSeries`)。
それぞれ `result` の 波形 / `SeriesRun` と 1 対 1。

**実行を知らない** = ここに書けるのは「何を回すか」だけで、どの surrogate で回すかも
回した結果も持たない (実行は `run`、結果は `result`)。おかげで**同一性が記述だけで
決まり** (`hash` を持つのは再利用の単位である `EvalSeries`)、
surrogate 層にも依存しない (`surrogate.model` がこの module を import する)。
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
from ..neurons import MCMODELS
from ._current_catalog import CURRENT_MAP


def _digest(key: dict) -> str:
    """dict → 短縮ハッシュ。鍵の作り方 (整列・既定の str 化) を 1 箇所に置く。"""
    return hashlib.sha1(
        json.dumps(key, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]


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
    # 注入ノード名。適用先の形態ではなく**実験の記述**なので、同じ target を部位だけ
    # 変えて回せる。既定は全モデル共通の soma 規約 (`neurons` の命名規約)。
    stim: str = "soma"

    def to_dict(self) -> dict:
        """永続化 (MLflow の学習 run / 評価 run) が仕様として持ち回る形へ。
        **名前だけを書く** = 適用先のグラフも電流の波形も保存しない (どちらも
        `MCMODELS` / `CURRENT_MAP` から名前で解ける導出物)。"""
        return {
            "target": self.target,
            "current_type": self.current_type,
            "dt": self.dt,
            "current_params": self.current_params,
            "stim": self.stim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            target=str(d["target"]),
            current_type=str(d["current_type"]),
            dt=float(d["dt"]),
            current_params=dict(d.get("current_params") or {}),
            stim=str(d["stim"]),
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
        """仕様 → 実行入力 (原系)。置換系は `Surrogate.apply` が
        この実行入力から非破壊で作る。

        **注入先の名前をここで index へ解く** = simulator には「どのノードへ
        いくら注入されるか」だけを見せる (電流は全モデル共通の密度 [μA/cm^2]
        規約なので換算は要らない)。"""
        return DatasetConfig(
            dt=self.dt,
            net=self.net,
            stim_idx=self.net.name_to_idx(self.stim),
            current=self.current(),
        )


@dataclass(frozen=True)
class EvalSeries:
    """**1 回の掃引実験の記述**: 何を (`spec`)・どこを置換して (`replace_targets`)・
    どの電流パラメータで振るか (`param`/`values`)。`param` を渡さなければ単発
    (点 1 つ) で、以降は掃引と同じ経路を通る。

    **どの surrogate で回すかも run 軸も持たない**: 置換器を掛けるのは `run`、
    run 軸に開いた結果を持つのは `result.SeriesResults` で、どちらもこの型の外に居る。
    掃引の記述が置換器から独立している = 原系の再利用が `hash` 1 本で効く。

    **保存の単位でもある**: 1 系列 = 1 評価 run (点列を丸ごと 1 artifact に持つ)
    なので、「同じ掃引を既に回したか」を引く鍵 (原系は `hash`、置換系は
    `replaced_hash`) と往復の形 (`to_dict` / `from_dict`) をこの型が持つ。
    """

    spec: SimSpec
    # 置換するノード名 (**明示指定**)。置換器そのものは知らないまま「適用先のどこを
    # 置換する実験か」だけを書く = 適用先の形態が変わっても置換範囲は動かない。
    # 互換かどうかは surrogate 側が名前ごとに検証する (`SurrogateSpec.applicable`)。
    replace_targets: tuple[str, ...]
    param: str | None = None  # 掃引する電流パラメータ名 (None=単発)。図の x 軸
    values: Sequence[float] = ()  # 掃引点の値列 (等間隔でなくてもよい)

    def to_dict(self) -> dict:
        """永続化 (評価 run の param) が持ち回る形 = 掃引の定義そのもの。"""
        return {
            "spec": self.spec.to_dict(),
            "replace_targets": list(self.replace_targets),
            "param": self.param,
            "values": [float(v) for v in self.values],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            spec=SimSpec.from_dict(d["spec"]),
            replace_targets=tuple(d["replace_targets"]),
            param=d["param"] or None,
            values=[float(v) for v in d["values"]],
        )

    def hash(self) -> str:
        """**原系の波形を既に回したか**の鍵。原系を決めるもの (適用先 × 電流 × 点列)
        だけで作り、**置換範囲も置換器も含まない** → 置換範囲だけが違う対照系列
        どうしでも原系 run を 1 本共有できる (原系は置換に依らないので正しい)。

        置換に効く要素を足したらここから外さない限り原系の共有が切れるので、
        除外は `replace_targets` を名指しで 1 箇所だけ書く。

        完全な仕様は波形 run 側が別に持つので、短縮ハッシュで足りる。"""
        return _digest(
            {k: v for k, v in self.to_dict().items() if k != "replace_targets"}
        )

    def replaced_hash(self) -> str:
        """**置換系の波形を既に回したか**の鍵 = 原系の鍵 + 置換範囲。置換器 (学習 run)
        だけは含まない — それは呼び出し側がここに組む (`mlflow_io.series`)。"""
        return _digest(
            {"original": self.hash(), "replace_targets": list(self.replace_targets)}
        )

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
