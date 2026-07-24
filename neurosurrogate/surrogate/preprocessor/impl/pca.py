"""PCA による gate ↔ latent 線形圧縮 preprocessor。"""

import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA

from ....core.opcost import OpCost
from ..base import Preprocessor


class PCAPreprocessor(Preprocessor):
    def __init__(
        self,
        components: np.ndarray,
        mean: np.ndarray,
        explained_variance: np.ndarray,
        explained_variance_ratio: np.ndarray,
        full_explained_variance_ratio: np.ndarray,
    ):
        self.components = components
        self.mean = mean
        self.explained_variance = explained_variance
        self.explained_variance_ratio = explained_variance_ratio
        # 全 (捨てた分も含む) 成分の寄与率。scree 図で n_components 選択の妥当性
        # (どこで累積が飽和するか) を見るために保持する。
        self.full_explained_variance_ratio = full_explained_variance_ratio

    @classmethod
    def fit(
        cls, train_gate: np.ndarray, n_components: int, spec: dict
    ) -> "PCAPreprocessor":
        # 全成分で 1 度 fit し上位 n を採る (2 度 fit を避ける)。full_* は捨てた
        # 成分の寄与率も含み、保持成分は先頭 n_components を切り出す。
        pca = PCA().fit(train_gate)
        inst = cls(
            components=np.asarray(pca.components_[:n_components]),
            mean=np.asarray(pca.mean_),
            explained_variance=np.asarray(pca.explained_variance_[:n_components]),
            explained_variance_ratio=np.asarray(
                pca.explained_variance_ratio_[:n_components]
            ),
            full_explained_variance_ratio=np.asarray(pca.explained_variance_ratio_),
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
        # 保持成分ごとの寄与率 (連番) + 累積 (= n_components でどれだけ説明できたか)。
        return {
            **{
                f"pca/explained_variance_ratio_{i + 1}": float(r)
                for i, r in enumerate(self.explained_variance_ratio)
            },
            "pca/cumulative_explained_variance_ratio": float(
                self.explained_variance_ratio.sum()
            ),
            "pca/reconstruction_mse": self.reconstruction_mse,
            "pca/reconstruction_mse_ratio": self.reconstruction_mse_ratio,
        }

    def opcost(self) -> OpCost:
        # decode: gate ごとに latent 数の積 + (latent-1 加算 + mean 1 加算)。
        n_latent, n_gates = self.components.shape
        return OpCost(mul=n_latent * n_gates, pm=n_latent * n_gates)
