"""PCA による gate ↔ latent 線形圧縮 preprocessor。"""

import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA

from ...core.opcost import OpCost
from .base import Preprocessor


class PCAPreprocessor(Preprocessor):
    def __init__(
        self,
        components: np.ndarray,
        mean: np.ndarray,
        explained_variance: np.ndarray,
        explained_variance_ratio: np.ndarray,
    ):
        self.components = components
        self.mean = mean
        self.explained_variance = explained_variance
        self.explained_variance_ratio = explained_variance_ratio

    @classmethod
    def fit(
        cls, train_gate: np.ndarray, n_components: int, spec: dict
    ) -> "PCAPreprocessor":
        pca = PCA(n_components=n_components).fit(train_gate)
        inst = cls(
            components=np.asarray(pca.components_),
            mean=np.asarray(pca.mean_),
            explained_variance=np.asarray(pca.explained_variance_),
            explained_variance_ratio=np.asarray(pca.explained_variance_ratio_),
        )
        inst._set_fit_artifacts(train_gate)
        return inst

    @property
    def n_features(self) -> int:
        return int(self.components.shape[1])

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.asarray((np.asarray(x) - self.mean) @ self.components.T)

    def decode(self, state: jnp.ndarray) -> jnp.ndarray:
        return state @ jnp.asarray(self.components) + jnp.asarray(self.mean)

    def metrics(self) -> dict:
        return {
            "pca/explained_variance_ratio": float(self.explained_variance_ratio[0]),
            "pca/explained_variance": float(self.explained_variance[0]),
            "pca/reconstruction_mse": self.reconstruction_mse,
            "pca/reconstruction_mse_ratio": self.reconstruction_mse_ratio,
        }

    def opcost(self) -> OpCost:
        # decode: gate ごとに latent 数の積 + (latent-1 加算 + mean 1 加算)。
        n_latent, n_gates = self.components.shape
        return OpCost(mul=n_latent * n_gates, pm=n_latent * n_gates)
