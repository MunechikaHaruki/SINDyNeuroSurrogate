"""評価対象の仕様 (設定 dict → 型) と結果のキー付け規約。marimo/mlflow 非依存。

**単発と掃引を分けない**: 評価 1 本 = 適用先 target × 電流 × (任意の) 掃引軸 で、
単発は「掃引軸なし = 点が 1 つ」の退化形にすぎない (掃引点 0/1 か複数かは軸の本数の
違いであって別種の実験ではない)。かつて `SimSpec`/`SweepSpec` に割れていたせいで
「掃引だけ run 軸を持つ」「単発だけ置換可否で除外される」といった食い違いが生えた。

ここが持つのは**計算入力だけ** (設定ファイル scripts/conf/eval.json: 1 entry =
適用先 × 電流 + 任意の `sweep` 軸)。描画の宣言 (`compare` / 表示設定) は別ファイル
(scripts/conf/draw.json) の関心なので `view.report` が型で持つ = **設定 dict を
domain 全体で持ち回らない** (パースは入口 1 回、以降は型で渡す)。

**実行は知らない** — 仕様 → 実行の向きに依存を張り、シミュを回すのは `eval.py`
(`run_evals`) 側。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
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
    """評価結果の run 軸の識別キー列 (与えた順)。

    `meta.label` は学習構造 + 学習データまでしか区別しない → library_specs 違いや
    同 config の再実行は同じ label になるため連番で潰れを防ぐ。
    """
    return dedupe_labels([s.meta.label for s in surrogates])


# --- 型: 掃引軸 + 評価仕様 ---------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SweepAxis:
    """掃引軸 = 電流パラメータ 1 つを等間隔に振る宣言。**この型の有無が単発と掃引の
    唯一の差**で、無ければ点が 1 つになるだけ。"""

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
class EvalSpec:
    """1 本の評価仕様 = 適用先 target × 電流 × (任意の) 掃引軸。dt も entry ごとに
    持つ (「設定全体で 1 つの電流/dt」は存在しない)。

    掃引軸が無ければ点は 1 つ = かつての単発。**単発と掃引で別の型を持たない**ので、
    実行も保存も描画も 1 経路で済む (点が 1 個か N 個かだけの違い)。
    """

    target: str
    current_type: str
    dt: float
    current_params: dict = field(default_factory=dict)
    name: str | None = None
    sweep: SweepAxis | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            target=str(d["target"]),
            current_type=str(d["current_type"]),
            dt=float(d["dt"]),
            current_params=dict(d.get("current_params") or {}),
            name=d.get("name"),
            sweep=SweepAxis.from_dict(d["sweep"]) if d.get("sweep") else None,
        )

    def to_dict(self) -> dict:
        """設定 entry と同じ形へ戻す (結果 artifact が入力仕様として持ち回る)。"""
        return {
            "target": self.target,
            "current_type": self.current_type,
            "dt": self.dt,
            "current_params": self.current_params,
            "name": self.name,
            "sweep": asdict(self.sweep) if self.sweep else None,
        }

    def key(self) -> str:
        """**同じ入力とみなす単位**の正規化文字列 (dict 順序に依らず一致する)。
        artifact の同一系列判定 (`ArtifactMeta.group_key`) や保存先 dir 名の
        hash 元など、「この spec と同じか」を問う側はここを経由する (json.dumps
        の正規化を持ち回り側で書き直させない)。"""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @property
    def label(self) -> str:
        """既定は適用先名。`name` 指定で電流の意図を名前に出せる。"""
        return self.name or self.target

    @property
    def net(self) -> NeuronGraph:
        """適用先の MC モデル。**target 名 → ネットの解決はここだけ**が行い、結果型も
        描画も spec 経由で引く (文字列キーを持ち回らない)。"""
        return MCMODELS[self.target]

    def replaceable(self, meta: SurrogateMeta) -> bool:
        """この surrogate をこの適用先へ置換できるか。**実行するかもエラー図を出すかも
        この 1 つの述語**で決める (欠落キーで非互換を伝えない)。"""
        return bool(replaced_names(meta, self.net))

    @property
    def points(self) -> list[float | None]:
        """掃引点の値 (掃引軸が無ければ `[None]` = 点 1 つ)。結果の点軸はこれと
        1 対 1 で、図の列見出しも掃引軸があるときだけ値を出す。"""
        return [float(v) for v in self.sweep.values] if self.sweep else [None]

    def dataset_at(self, point: float | None) -> DatasetConfig:
        """点 1 つ分の入力 (原系)。置換系は `apply_surrogate` が非破壊で作る。
        単発も掃引も**同じ構築経路**を通る (掃引だけ別の組み方をしない)。"""
        params = self.current_params
        if point is not None and self.sweep:
            params = {**params, self.sweep.param: point}
        return DatasetConfig.build_dataset(
            model_name=self.target,
            dt=self.dt,
            current_type=self.current_type,
            current_params=params,
        )

    def dataset(self) -> DatasetConfig:
        """代表点 (先頭) の入力。電流プレビューのように 1 本だけ要る用途向け。"""
        return self.dataset_at(self.points[0])


# --- 設定 dict ⇄ 型 (唯一の変換口) -------------------------------------------------


def parse_evals(entries: list[dict]) -> dict[str, EvalSpec]:
    """`eval.json` (entry の配列) → label → EvalSpec (**dict を型へ落とす唯一の
    入口**)。同じ label が複数あれば連番で潰れを防ぐ (`run_labels` と同規約) = 結果
    dict / 図名のキーはここが単一源。"""
    specs = [EvalSpec.from_dict(d) for d in entries]
    return dict(zip(dedupe_labels([s.label for s in specs]), specs, strict=True))


def usable(meta: SurrogateMeta, specs: dict[str, EvalSpec]) -> bool:
    """宣言した評価の 1 本でも置換できる surrogate か = UI の run 絞り込み条件。"""
    return any(s.replaceable(meta) for s in specs.values())
