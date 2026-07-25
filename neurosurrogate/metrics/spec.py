"""評価対象の仕様 (設定 dict → 型) と結果のキー付け規約。marimo/mlflow 非依存。

設定ファイル (scripts/conf/base.json) は `sim`/`sweep` の 2 セクションを持ち、
どちらも entry の list、1 entry = **適用先 target × 電流** を自己完結で持つ
(sweep はさらに掃引軸)。**そのスキーマを知るのはこの module だけ** で、UI 層は
生の dict を渡すだけ: どのキーが何を意味するか・結果をどのキーで束ねるかはここが
決める (表示設定 `draw` は widget 由来なので UI 側)。

**実行は知らない** — 仕様 → 実行の向きに依存を張り、シミュを回すのは `eval.py`
(`run_sims` / `run_sweeps`) 側。
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Self, TypeVar

import numpy as np

from ..core.network import DatasetConfig, NeuronGraph
from ..neurons import MCMODELS
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.meta import SurrogateMeta
from ..surrogate.replace import replaced_names


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


def sweep_labels(surrogates: list[SurrogateBundle]) -> list[str]:
    """掃引結果の run 軸の識別キー列 (与えた順)。

    `meta.label` は学習構造 + 学習データまでしか区別しない → library_specs 違いや
    同 config の再実行は同じ label になるため連番で潰れを防ぐ。
    """
    return dedupe_labels([s.meta.label for s in surrogates])


@dataclass(frozen=True, kw_only=True)
class SimSpec:
    """1 本の置換シミュ仕様 = 適用先 target × 電流。dt も entry ごとに持つ
    (「設定全体で 1 つの電流/dt」は存在しない)。"""

    target: str
    current_type: str
    dt: float
    current_params: dict = field(default_factory=dict)
    name: str | None = None

    @classmethod
    def _fields_from(cls, d: dict) -> dict:
        """設定 entry → コンストラクタ引数。サブクラスは自分の軸を足す。"""
        return {
            "target": str(d["target"]),
            "current_type": str(d["current_type"]),
            "dt": float(d["dt"]),
            "current_params": dict(d.get("current_params") or {}),
            "name": d.get("name"),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(**cls._fields_from(d))

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

    def _dataset(self, current_params: dict) -> DatasetConfig:
        return DatasetConfig.build_dataset(
            model_name=self.target,
            dt=self.dt,
            current_type=self.current_type,
            current_params=current_params,
        )

    def dataset(self) -> DatasetConfig:
        """このシミュの入力 (原系)。置換系は `apply_surrogate` が非破壊で作る。"""
        return self._dataset(self.current_params)


@dataclass(frozen=True, kw_only=True)
class SweepSpec(SimSpec):
    """SimSpec に掃引軸を足したもの。current_params は掃引軸以外の固定値になる。"""

    sweep_param: str
    amp_start: float
    amp_stop: float
    amp_steps: int

    @classmethod
    def _fields_from(cls, d: dict) -> dict:
        return {
            **super()._fields_from(d),
            "sweep_param": str(d["sweep_param"]),
            "amp_start": float(d["amp_start"]),
            "amp_stop": float(d["amp_stop"]),
            "amp_steps": int(d["amp_steps"]),
        }

    @property
    def amp_values(self) -> np.ndarray:
        return np.linspace(self.amp_start, self.amp_stop, self.amp_steps)

    def dataset_at(self, amp: float) -> DatasetConfig:
        """掃引 1 点の入力。単発と同じ構築経路を通す (掃引だけ別の組み方をしない)。"""
        return self._dataset({**self.current_params, self.sweep_param: float(amp)})


S = TypeVar("S", bound=SimSpec)


def _labeled(specs: list[S]) -> dict[str, S]:
    """label → spec。同じ label が複数あれば連番で潰れを防ぐ (`sweep_labels` と同規約)。
    結果 dict / 図名のキーはここが単一源 = 計算と描画で食い違わない。"""
    return dict(zip(dedupe_labels([s.label for s in specs]), specs, strict=True))


def parse_sims(cfg: dict) -> dict[str, SimSpec]:
    """設定 dict の `sim` セクション → label → SimSpec。"""
    return _labeled([SimSpec.from_dict(d) for d in cfg.get("sim", [])])


def parse_sweeps(cfg: dict) -> dict[str, SweepSpec]:
    """設定 dict の `sweep` セクション → label → SweepSpec。"""
    return _labeled([SweepSpec.from_dict(d) for d in cfg.get("sweep", [])])


def cfg_specs(cfg: dict) -> list[SimSpec]:
    """設定が宣言する全 spec (sim + sweep)。"""
    return [*parse_sims(cfg).values(), *parse_sweeps(cfg).values()]


def usable(meta: SurrogateMeta, cfg: dict) -> bool:
    """この設定で 1 本でも置換シミュできる surrogate か = UI の run 絞り込み条件。"""
    return any(s.replaceable(meta) for s in cfg_specs(cfg))
