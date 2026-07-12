from dataclasses import dataclass
from typing import Any

import numpy as np
import pysindy as ps


@dataclass
class PCABundle:
    components: np.ndarray
    mean: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    reconstruction_mse: float
    reconstruction_mse_ratio: float

    @classmethod
    def from_preprocessor(cls, preprocessor, train_gate: np.ndarray) -> "PCABundle":
        reconstructed = preprocessor.inverse_transform(
            preprocessor.transform(train_gate)
        )
        mse = float(np.mean((train_gate - reconstructed) ** 2))
        return cls(
            components=np.asarray(preprocessor.components_),
            mean=np.asarray(preprocessor.mean_),
            explained_variance=np.asarray(preprocessor.explained_variance_),
            explained_variance_ratio=np.asarray(preprocessor.explained_variance_ratio_),
            reconstruction_mse=mse,
            reconstruction_mse_ratio=mse / float(np.var(train_gate)),
        )


@dataclass
class PreprocessorBundle:
    preprocessor: Any
    bundle: PCABundle | None
    gate_inits: list


@dataclass
class SINDyBundle:
    xi: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    equations: str
    library_specs: list[dict]

    @classmethod
    def from_sindy(
        cls, sindy: ps.SINDy, target_names: list[str], library_specs: list[dict]
    ) -> "SINDyBundle":
        return cls(
            xi=sindy.coefficients(),
            feature_names=sindy.get_feature_names(),
            target_names=target_names,
            equations="\n".join(sindy.equations(precision=3)),
            library_specs=library_specs,
        )
