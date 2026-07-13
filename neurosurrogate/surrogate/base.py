from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import ClassVar

import joblib

from ..core.network import Compartment, CompartmentType, DatasetConfig
from ..core.opcost import OpCost
from ..core.simulator import unified_simulator
from .bundle import PreprocessorBundle, SINDyBundle

BUNDLE_FILE = "surrogate.joblib"


class Verdict(Enum):
    """サロゲート置換の妥当性判定 (学習ドメインとの照合結果)。"""

    REPLACE = auto()  # 型一致 かつ params 一致 → 置換
    MISMATCH = auto()  # 型一致 だが params 不一致 → 疑わしい (置換不可)
    SKIP = auto()  # 型不一致 → 無関係 (対象外)


@dataclass(frozen=True)
class SurrogateMeta:
    surrogate_type: str
    dataset: DatasetConfig
    train_comp_id: int

    @classmethod
    def build(
        cls, surrogate_type: str, datasets: dict, train_comp_identifier: str
    ) -> "SurrogateMeta":
        dataset = DatasetConfig.build_dataset(**datasets)
        return cls(
            surrogate_type=surrogate_type,
            dataset=dataset,
            train_comp_id=dataset.net.name_to_idx(train_comp_identifier),
        )

    @property
    def train_comp(self) -> Compartment:
        return self.dataset.net.nodes[self.train_comp_id]

    @property
    def train_comp_type(self) -> CompartmentType:
        """学習元コンパートメントの物理型 (= 置換対象)。"""
        return self.train_comp.type

    def verdict(self, comp: Compartment) -> Verdict:
        """comp が学習ドメイン (train_comp の type+params) に属すか判定。

        型が違えば無関係 (SKIP)、型は同じだが params が違えば疑わしい
        (MISMATCH)、両方一致で置換可 (REPLACE)。
        """
        train = self.train_comp
        if comp.type != train.type:
            return Verdict.SKIP
        if comp.params != train.params:
            return Verdict.MISMATCH
        return Verdict.REPLACE

    def replaceables(self, dataset: DatasetConfig) -> set[str]:
        """dataset 内の置換対象ノード名を返す (fail first)。

        - 型一致・params 不一致 (MISMATCH) が1つでもあれば即エラー
        - 置換対象 (REPLACE) が皆無なら即エラー (モデルとデータが噛み合わず)
        """
        verdicts = {n.name: self.verdict(n) for n in dataset.net.nodes}
        train = self.train_comp

        mismatched = [
            n for n in dataset.net.nodes if verdicts[n.name] is Verdict.MISMATCH
        ]
        if mismatched:
            raise ValueError(
                f"型 {train.type.name!r} 一致だが params 不一致のノード "
                f"{[n.name for n in mismatched]}: サロゲートは学習 params 専用。\n"
                f"  train({train.name}): {train.params}\n"
                + "\n".join(f"  node({n.name}): {n.params}" for n in mismatched)
            )
        targets = {name for name, v in verdicts.items() if v is Verdict.REPLACE}
        if not targets:
            raise ValueError(
                f"学習型 {train.type.name!r} のノードが dataset "
                f"{dataset.model_name!r} に存在しない → 置換対象ゼロ。適用不可"
            )
        return targets

    @property
    def original_opcost(self) -> OpCost | None:
        return self.train_comp.type.opcost

    def simulate(self):
        return unified_simulator(self.dataset)


class NeuroSurrogateBase(ABC):
    SURROGATE_TYPE: ClassVar[str]
    _meta: SurrogateMeta
    _sindy_bundle: SINDyBundle
    _preprocessor_bundle: PreprocessorBundle

    def __init__(self, datasets: dict, train_comp_identifier: str):
        self._meta = SurrogateMeta.build(
            surrogate_type=self.SURROGATE_TYPE,
            datasets=datasets,
            train_comp_identifier=train_comp_identifier,
        )
        self._train_xr = self._meta.simulate()

    @classmethod
    def build(cls, type: str, init: dict) -> "NeuroSurrogateBase":
        from . import SURR_CLS

        return SURR_CLS[type](**init)

    @property
    def meta(self) -> SurrogateMeta:
        return self._meta

    @abstractmethod
    def fit(self, preprocessor, optimizer, library_specs: list[dict]) -> None: ...

    @property
    @abstractmethod
    def surr_comp_type(self) -> CompartmentType:
        """置換後の surrogate CompartmentType (学習結果から構築)。"""
        ...

    def apply(self, dataset: DatasetConfig) -> DatasetConfig:
        """学習ドメインに属す全ノードを surrogate に置換 (検証は meta が担う)。"""
        targets = self.meta.replaceables(dataset)
        return dataset.with_surrogate(
            self.surr_comp_type, accept=lambda n: n.name in targets
        )

    @property
    @abstractmethod
    def opcost(self) -> OpCost: ...

    @property
    def sindy_bundle(self) -> SINDyBundle:
        return self._sindy_bundle

    @property
    def preprocessor_bundle(self) -> PreprocessorBundle:
        return self._preprocessor_bundle

    def _set_bundles(
        self,
        sindy_bundle: SINDyBundle,
        preprocessor_bundle: PreprocessorBundle,
    ) -> None:
        self._sindy_bundle = sindy_bundle
        self._preprocessor_bundle = preprocessor_bundle

    def metrics(self) -> dict:
        return {
            **self.sindy_bundle.xi_metrics(),
            **self.preprocessor_bundle.metrics(),
            **self.opcost.diff_dict(self._meta.original_opcost),
        }

    def save(self, dir: Path | str) -> None:
        joblib.dump(
            {
                "meta": self._meta,
                "sindy_bundle": self.sindy_bundle,
                "preprocessor_bundle": self.preprocessor_bundle,
            },
            Path(dir) / BUNDLE_FILE,
        )
