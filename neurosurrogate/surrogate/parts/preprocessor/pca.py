"""PCA による gate ↔ latent 線形圧縮 preprocessor。"""

import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA

from ....core.opcost import OpCost
from .. import Preprocessor
from .fit_artifacts import fit_artifacts


class PCAPreprocessor(Preprocessor):
    # gate_inits と並んで fit 末尾で埋まる。契約に載らない (metrics でしか読まない)
    # ので実装側で宣言する。
    reconstruction: dict[str, float]

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
            **{f"pca/{k}": v for k, v in self.reconstruction.items()},
        }

    def opcost(self) -> OpCost:
        # decode: gate ごとに latent 数の積 + (latent-1 加算 + mean 1 加算)。
        n_latent, n_gates = self.components.shape
        return OpCost(mul=n_latent * n_gates, pm=n_latent * n_gates)


def fit_pca(train_gates: list[np.ndarray], n_components: int) -> PCAPreprocessor:
    """comp ごとの学習ゲート軌道から線形圧縮を学習する (**preprocessor 側の入口**)。

    軌道を跨いで縦連結するのはここ: 時間の並びが意味を持つのは微分を取る同定側で、
    座標変換にとっては点の集合でしかない。hyperparams は無いので引数もこれだけ =
    preprocessor ブロックに何か書けば TypeError (黙って無視しない)。
    """
    train_gate = np.concatenate(train_gates, axis=0)
    # 全成分で 1 度 fit し上位 n を採る (2 度 fit を避ける)。full_* は捨てた
    # 成分の寄与率も含み、保持成分は先頭 n_components を切り出す。
    pca = PCA().fit(train_gate)
    inst = PCAPreprocessor(
        components=np.asarray(pca.components_[:n_components]),
        mean=np.asarray(pca.mean_),
        explained_variance=np.asarray(pca.explained_variance_[:n_components]),
        explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_[:n_components]
        ),
        full_explained_variance_ratio=np.asarray(pca.explained_variance_ratio_),
    )
    inst.gate_inits, inst.reconstruction = fit_artifacts(inst, train_gate)
    return inst
