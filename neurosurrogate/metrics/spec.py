"""評価する対象の仕様 (設定 dict → 型) とその実行。marimo/mlflow 非依存。

設定ファイル (scripts/conf/base.json) は `sim`/`sweep` の 2 セクションを持ち、
どちらも entry の list、1 entry = **適用先 target × 電流** を自己完結で持つ
(sweep はさらに掃引軸)。**そのスキーマを知るのはこの module だけ** で、UI 層は
生の dict を渡すだけ: どのキーが何を意味するか・何回どの順でシミュするか・結果を
どのキーで束ねるかは全部ここが決める (表示設定 `draw` は widget 由来なので UI 側)。
"""

from dataclasses import dataclass, field
from typing import Self, TypeVar

from ..core.network import DatasetConfig
from ..neurons import MCMODELS
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.replace import replaced_names
from .eval import EvalResult, evaluate
from .eval_sweep import (
    CurrentSweepConfig,
    SweepEval,
    dedupe_labels,
    evaluate_sweep,
    sweep_labels,
)


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
    def _common(cls, d: dict) -> dict:
        return {
            "target": str(d["target"]),
            "current_type": str(d["current_type"]),
            "dt": float(d["dt"]),
            "current_params": dict(d.get("current_params") or {}),
            "name": d.get("name"),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(**cls._common(d))

    @property
    def label(self) -> str:
        """既定は適用先名。`name` 指定で電流の意図を名前に出せる。"""
        return self.name or self.target

    def dataset(self) -> DatasetConfig:
        """このシミュの入力 (原系)。置換系は `apply_surrogate` が非破壊で作る。"""
        return DatasetConfig.build_dataset(
            model_name=self.target,
            dt=self.dt,
            current_type=self.current_type,
            current_params=self.current_params,
        )


@dataclass(frozen=True, kw_only=True)
class SweepSpec(SimSpec):
    """SimSpec に掃引軸を足したもの。current_params は掃引軸以外の固定値になる。"""

    sweep_param: str
    amp_start: float
    amp_stop: float
    amp_steps: int

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            **cls._common(d),
            sweep_param=str(d["sweep_param"]),
            amp_start=float(d["amp_start"]),
            amp_stop=float(d["amp_stop"]),
            amp_steps=int(d["amp_steps"]),
        )

    def sweep_config(self) -> CurrentSweepConfig:
        return CurrentSweepConfig(
            current_type=self.current_type,
            sweep_param=self.sweep_param,
            amp_start=self.amp_start,
            amp_stop=self.amp_stop,
            amp_steps=self.amp_steps,
            base_params=self.current_params,
        )


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


def cfg_targets(cfg: dict) -> list[str]:
    """設定が宣言する適用先モデル名 (sim + sweep、重複除去・出現順)。UI の run 絞り込み
    と comp 選択肢の母集合。"""
    specs: list[SimSpec] = [*parse_sims(cfg).values(), *parse_sweeps(cfg).values()]
    return list(dict.fromkeys(s.target for s in specs))


def run_sims(
    bundle: SurrogateBundle,
    specs: dict[str, SimSpec],
) -> dict[str, EvalResult]:
    """spec ごとに原系/置換系を並走シミュし label → EvalResult。置換可能な comp が
    無い spec (非互換な target) は simulate せず落とす (呼び出し側が欠落で判別)。"""
    return {
        label: evaluate(bundle, s.dataset())
        for label, s in specs.items()
        if replaced_names(bundle.meta, MCMODELS[s.target])
    }


def run_sweeps(
    bundles: list[SurrogateBundle],
    specs: dict[str, SweepSpec],
) -> dict[str, SweepEval]:
    """spec ごとに amp 掃引評価し label → SweepEval。掃引結果は **entry 軸 (この dict)
    × run 軸 (`sweep_labels`)** の 2 段。"""
    return {
        label: evaluate_sweep(
            dict(zip(sweep_labels(bundles), bundles, strict=True)),
            model_name=s.target,
            dt=s.dt,
            cfg=s.sweep_config(),
        )
        for label, s in specs.items()
    }
