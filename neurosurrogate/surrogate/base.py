from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import joblib
import numpy as np

from ..core.coords import StateAccumulator, set_coords
from ..core.network import Compartment, DatasetConfig
from ..core.opcost import OpCost
from ..core.simulator import unified_simulator
from .bundle import PreprocessorBundle, SINDyBundle

_BUNDLE_FILE = "surrogate.joblib"


@dataclass(frozen=True)
class SurrogateMeta:
    surrogate_type: str
    dataset: DatasetConfig
    train_comp_id: int

    def to_dict(self) -> dict:
        return {
            "surrogate_type": self.surrogate_type,
            "dataset": self.dataset.to_dict(),
            "train_comp_id": self.train_comp_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SurrogateMeta":
        return cls(
            surrogate_type=d["surrogate_type"],
            dataset=DatasetConfig.from_dict(d["dataset"]),
            train_comp_id=d["train_comp_id"],
        )


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
    _sindy_bundle: SINDyBundle
    _preprocessor_bundle: PreprocessorBundle

    def __init__(self, datasets: dict, train_comp_identifier: str):
        self._dataset = DatasetConfig.build_dataset(**datasets)
        self.train_comp_id: int = self._dataset.net.name_to_idx(train_comp_identifier)
        self._train_xr = unified_simulator(self._dataset)

    def fetch_meta(self) -> SurrogateMeta:
        return SurrogateMeta(
            surrogate_type=self.SURROGATE_TYPE,
            dataset=self._dataset,
            train_comp_id=self.train_comp_id,
        )

    @abstractmethod
    def fit(self, preprocessor, optimizer, library_specs: list[dict]) -> None: ...

    @abstractmethod
    def make_surr_comp(self, name: str, **kwargs) -> Compartment: ...

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

    def metrics(self, meta: "SurrogateMeta") -> dict:
        original_opcost = meta.dataset.net.nodes[meta.train_comp_id].type.opcost
        return {
            **self.sindy_bundle.xi_metrics(),
            **self.preprocessor_bundle.metrics(),
            **self.opcost.diff_dict(original_opcost),
        }

    def save(self, dir: Path | str) -> None:
        joblib.dump(
            {
                "sindy_bundle": self.sindy_bundle,
                "preprocessor_bundle": self.preprocessor_bundle,
            },
            Path(dir) / _BUNDLE_FILE,
        )

    @classmethod
    def load(cls, dir: Path | str) -> Self:
        self = cls.__new__(cls)
        self._set_bundles(**joblib.load(Path(dir) / _BUNDLE_FILE))
        return self
