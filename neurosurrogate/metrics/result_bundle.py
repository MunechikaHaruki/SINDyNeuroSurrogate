from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pysindy as ps

from .opcost import OpCost

if TYPE_CHECKING:
    from collections.abc import Callable


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

    def metrics(self) -> dict:
        return {
            "pca/explained_variance_ratio": float(self.explained_variance_ratio[0]),
            "pca/explained_variance": float(self.explained_variance[0]),
            "pca/reconstruction_mse": self.reconstruction_mse,
            "pca/reconstruction_mse_ratio": self.reconstruction_mse_ratio,
        }


@dataclass
class PreprocessorBundle:
    preprocessor: Any
    bundle: PCABundle | None
    gate_inits: list

    def metrics(self) -> dict:
        return self.bundle.metrics() if self.bundle is not None else {}


@dataclass
class SINDyBundle:
    xi: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    equations: str
    library_specs: list[dict]

    @classmethod
    def from_sindy(
        cls,
        library_specs: list[dict],
        optimizer,
        x: np.ndarray,
        u: np.ndarray,
        t: np.ndarray,
        target_names: list[str],
        input_names: list[str],
    ) -> "SINDyBundle":
        from ..core.libraries import FeatureLibrary

        sindy = ps.SINDy(
            feature_library=FeatureLibrary.build(library_specs).library,
            optimizer=optimizer,
        )
        sindy.fit(x, u=u, t=t, feature_names=target_names + input_names)
        return cls(
            xi=sindy.coefficients(),
            feature_names=sindy.get_feature_names(),
            target_names=target_names,
            equations="\n".join(sindy.equations(precision=3)),
            library_specs=library_specs,
        )

    def xi_metrics(self) -> dict[str, float]:
        nnz = int((self.xi != 0).sum())
        return {"nnz": nnz, "nnz_ratio": nnz / self.xi.size}

    def compute_theta(self) -> "Callable":
        from ..core.libraries import FeatureLibrary

        subs = FeatureLibrary.build(self.library_specs).sub_libraries

        def compute_theta(*inputs):
            values = []
            for sub in subs:
                bound = [inputs[i] for i in sub.inputs]
                for entry in sub.entries:
                    values.append(entry.func(*bound))
            return jnp.array(values, dtype=jnp.float64)

        return compute_theta

    def opcost(self) -> OpCost:
        from ..core.libraries import FeatureLibrary

        cost_map = FeatureLibrary.build(self.library_specs).to_base_cost(
            self.target_names + ["u"]
        )
        active_mask = np.any(self.xi != 0, axis=0)
        active_features = [
            f for i, f in enumerate(self.feature_names) if active_mask[i]
        ]
        nnz = np.count_nonzero(self.xi).item()
        cost = OpCost(mul=nnz, pm=max(0, nnz - int(self.xi.shape[0])))
        for feature in active_features:
            if feature not in cost_map:
                raise ValueError(f"Found Unknown base func: '{feature}'")
            cost = cost + cost_map[feature]
        return cost
