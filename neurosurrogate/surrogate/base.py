from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np

from ..core.coords import StateAccumulator, set_coords
from ..core.network import Compartment, CompartmentType, DatasetConfig
from ..core.opcost import OpCost
from ..core.simulator import unified_simulator
from .bundle import PreprocessorBundle, SINDyBundle

BUNDLE_FILE = "surrogate.joblib"


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

    @property
    def original_opcost(self) -> OpCost | None:
        return self.train_comp.type.opcost

    def simulate(self):
        return unified_simulator(self.dataset)


def get_gate_numpy(train_xr, target_comp_id):
    return train_xr["vars"].sel(gate=True, comp_id=target_comp_id).to_numpy()


def transform_gate(preprocessor, xr_data, target_comp_id):
    transformed_gate = preprocessor.transform(get_gate_numpy(xr_data, target_comp_id))
    n_latent = transformed_gate.shape[1]

    return set_coords(
        raw=np.concatenate(
            (
                xr_data["vars"]
                .sel(gate=False, comp_id=target_comp_id)
                .to_numpy()
                .reshape(-1, 1),
                transformed_gate,
            ),
            axis=1,
        ),
        u=xr_data["I_internal"].sel(node_id=target_comp_id).to_numpy(),
        coords=StateAccumulator(
            comp_id=[target_comp_id] * (n_latent + 1),
            variable=["V"] + [f"latent{i + 1}" for i in range(n_latent)],
            gate=[False] + [True] * n_latent,
        ).to_coords(),
        dt=float(xr_data.time[1] - xr_data.time[0]),
    )


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
