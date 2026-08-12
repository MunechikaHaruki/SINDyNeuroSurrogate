from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
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
    `eval.simulate` の引数、出所は結果側 (`eval.SimResult`)。おかげで `hash()` が
    「同じ波形を出す入力か」と正確に一致する。
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

    def key(self) -> str:
        """**同じ入力とみなす単位**の正規化文字列 (dict 順序に依らず一致する)。
        「この spec と同じか」を問う側はここを経由する。"""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    def hash(self) -> str:
        """`key()` の短縮ハッシュ。保存側が「同じ入力を既に回したか」を引くための
        キー (完全な仕様は別に持つので、衝突しない長さがあれば足りる)。"""
        return short_hash(self.key())

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


def short_hash(key: str) -> str:
    """正規化文字列 → 短縮ハッシュ。「同じものを既に回したか」を引く鍵の作り方を
    `SimSpec` と `eval.EvalSeries` で揃える (どちらも完全な仕様を別に持つので
    短くてよい)。"""
    return hashlib.sha1(key.encode()).hexdigest()[:8]
