"""評価対象の仕様 (設定 dict → 型) と結果のキー付け規約。marimo/mlflow 非依存。

**基本単位は 1 回のシミュレーション**: `SimSpec` は掃引点も run (surrogate) も
どちらも自身のフィールドとして持つ (どちらも「どのシミュを指すか」を決める同格の
パラメータ)。掃引軸は設定ファイル側の宣言 (`SweepAxis`) が `parse_evals` の時点で
`steps` 本の `SimSpec` へ展開し尽くす — 展開後の型に「掃引軸」という概念は残らない。
run は `run_id` (`None` = 原系) を持つだけの対称なフィールド。

**ここが持つのは計算入力だけ** (設定ファイル `scripts/conf/eval.json`: label →
entry の dict。1 entry = 適用先 target × 電流 + 任意の `sweep` 軸)。描画の宣言
(`compare` / 表示設定) は別ファイル (`scripts/conf/draw.json`) の関心なので
`metrics.report` が型で持つ = **設定 dict を domain 全体で持ち回らない** (パースは
入口 1 回、以降は型で渡す)。

**実行は知らない** — 仕様 → 実行の向きに依存を張り、シミュを回すのは `eval.run` 側。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Self

import numpy as np

from ..core.network import DatasetConfig, NeuronGraph
from ..neurons import MCMODELS
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.meta import SurrogateMeta
from ..surrogate.replace import replaced_names

# --- 結果 dict のラベル規約 -------------------------------------------------------


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


# --- 型: 掃引軸 + 評価仕様 ---------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SweepAxis:
    """掃引軸 = 電流パラメータ 1 つを等間隔に振る宣言。`parse_evals` が展開し尽くす
    ので、展開後の `SimSpec` はこの型を持たない (`sweep_param` という軸名だけ残る)。"""

    param: str
    start: float
    stop: float
    steps: int

    @classmethod
    def from_dict(cls, d: dict) -> SweepAxis:
        return cls(
            param=str(d["param"]),
            start=float(d["start"]),
            stop=float(d["stop"]),
            steps=int(d["steps"]),
        )

    @property
    def values(self) -> np.ndarray:
        return np.linspace(self.start, self.stop, self.steps)


@dataclass(frozen=True, kw_only=True)
class SimSpec:
    """1 回のシミュレーションの仕様 = 適用先 target × 電流 (掃引点は
    `current_params` に確定済み) × run。原系/置換系を非対称に扱わない —
    `run_id=None` が原系、それ以外が surrogate の MLflow run_id で、経路は共通。
    """

    name: str  # eval.json のキー = 掃引しても不変な系列名 (図の系列識別に使う)
    target: str
    current_type: str
    dt: float
    current_params: dict = field(default_factory=dict)
    sweep_param: str | None = None  # 掃引軸名 (None=掃引なし)。図の x 軸ラベル用
    run_id: str | None = None  # None=原系、str=surrogate の MLflow run_id

    @property
    def sweep_value(self) -> float | None:
        """掃引軸の値 (掃引なしなら None)。`current_params` に確定済みの値を読むだけ
        — 二重に持たない。"""
        return (
            float(self.current_params[self.sweep_param]) if self.sweep_param else None
        )

    def to_dict(self) -> dict:
        """artifact が入力仕様として持ち回る形へ。"""
        return {
            "name": self.name,
            "target": self.target,
            "current_type": self.current_type,
            "dt": self.dt,
            "current_params": self.current_params,
            "sweep_param": self.sweep_param,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            name=str(d["name"]),
            target=str(d["target"]),
            current_type=str(d["current_type"]),
            dt=float(d["dt"]),
            current_params=dict(d.get("current_params") or {}),
            sweep_param=d.get("sweep_param"),
            run_id=d.get("run_id"),
        )

    def key(self) -> str:
        """**同じ入力とみなす単位**の正規化文字列 (dict 順序に依らず一致する)。
        「この spec と同じか」を問う側はここを経由する。"""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    def hash(self) -> str:
        """`key()` の短縮ハッシュ。保存側が「同じ入力を既に回したか」を引くための
        キー (完全な仕様は別に持つので、衝突しない長さがあれば足りる)。"""
        return hashlib.sha1(self.key().encode()).hexdigest()[:8]

    @property
    def net(self) -> NeuronGraph:
        """適用先の MC モデル。**target 名 → ネットの解決はここだけ**が行い、結果型も
        描画も spec 経由で引く (文字列キーを持ち回らない)。"""
        return MCMODELS[self.target]

    def dataset(self) -> DatasetConfig:
        """入力 (原系)。置換系は `apply_surrogate` が非破壊で作る。"""
        return DatasetConfig.build_dataset(
            model_name=self.target,
            dt=self.dt,
            current_type=self.current_type,
            current_params=self.current_params,
        )


# --- 設定 dict ⇄ 型 (唯一の変換口) -------------------------------------------------


def _expand_sweep(name: str, d: dict) -> dict[str, SimSpec]:
    """1 entry (dict) → 掃引軸があれば `steps` 本、無ければ 1 本の `label → SimSpec`。
    label は掃引ありなら `f"{name}#{i}"`、無しなら `name` そのもの (dict 形式の
    `eval.json` はキー衝突が構造的に起きないので dedupe は要らない)。"""
    base_params = dict(d.get("current_params") or {})
    sweep = SweepAxis.from_dict(d["sweep"]) if d.get("sweep") else None
    if sweep is None:
        spec = SimSpec(
            name=name,
            target=str(d["target"]),
            current_type=str(d["current_type"]),
            dt=float(d["dt"]),
            current_params=base_params,
        )
        return {name: spec}
    specs = {}
    for i, v in enumerate(sweep.values):
        params = {**base_params, sweep.param: float(v)}
        specs[f"{name}#{i}"] = SimSpec(
            name=name,
            target=str(d["target"]),
            current_type=str(d["current_type"]),
            dt=float(d["dt"]),
            current_params=params,
            sweep_param=sweep.param,
        )
    return specs


def parse_evals(entries: dict[str, dict]) -> dict[str, SimSpec]:
    """`eval.json` (label → entry の dict) → label → SimSpec (**dict を型へ落とす
    唯一の入口**)。掃引軸がある entry は `steps` 本の SimSpec へ展開する
    (label は `f"{name}#{i}"`)。"""
    specs: dict[str, SimSpec] = {}
    for name, d in entries.items():
        specs.update(_expand_sweep(name, d))
    return specs


def usable(meta: SurrogateMeta, specs: dict[str, SimSpec]) -> bool:
    """宣言した評価の 1 本でも置換できる surrogate か = UI の run 絞り込み条件。"""
    return any(bool(replaced_names(meta, s.net)) for s in specs.values())
