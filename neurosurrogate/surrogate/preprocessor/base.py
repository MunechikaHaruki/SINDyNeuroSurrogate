"""ゲート ↔ 潜在の可逆変換を担う preprocessor 抽象基底。

fit 時に学習し、以降 encode/decode/metrics/opcost を **np パラメータ保持で自己完結**
する (生の学習オブジェクトは持たず直列化可能)。PCA/AE 実装は pca.py/autoencoder.py。
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import jax.numpy as jnp
import numpy as np

from ...core.opcost import OpCost


def _reconstruction_stats(
    encode: Callable, decode: Callable, train_gate: np.ndarray
) -> tuple[float, float]:
    reconstructed = np.asarray(decode(encode(train_gate)))
    mse = float(np.mean((train_gate - reconstructed) ** 2))
    return mse, mse / float(np.var(train_gate))


class Preprocessor(ABC):
    """ゲート ↔ 潜在の可逆変換。学習結果を np で保持し直列化可能。"""

    kind: ClassVar[str]
    # 以下は _set_fit_artifacts が fit 末尾で設定する (__init__ 引数ではない)。
    reconstruction_mse: float
    reconstruction_mse_ratio: float
    # 学習データ先頭の潜在 = 置換シミュの初期ゲート値。
    gate_inits: list

    @classmethod
    @abstractmethod
    def fit(
        cls, train_gate: np.ndarray, n_components: int, spec: dict
    ) -> "Preprocessor":
        """潜在次元 n_components (全種共通) と spec (種別固有 hyperparams) で学習。"""
        ...

    @abstractmethod
    def encode(self, x: np.ndarray) -> np.ndarray:
        """ゲート → 潜在 (診断 / 学習データ変換)。"""
        ...

    @abstractmethod
    def decode(self, state: jnp.ndarray) -> jnp.ndarray:
        """潜在 → ゲート (kernel で毎ステップ呼ぶ)。"""
        ...

    @abstractmethod
    def metrics(self) -> dict: ...

    @abstractmethod
    def opcost(self) -> OpCost:
        """decode 1 回の演算コスト (hybrid kernel の decode 分)。"""
        ...

    @property
    @abstractmethod
    def n_features(self) -> int:
        """encode 入力のゲート数 (transform_gate の幅整合用)。"""
        ...

    def _set_fit_artifacts(self, train_gate: np.ndarray) -> None:
        """encode/decode 確定後に再構成統計と初期潜在を埋める (fit 末尾で呼ぶ)。"""
        self.reconstruction_mse, self.reconstruction_mse_ratio = _reconstruction_stats(
            self.encode, self.decode, train_gate
        )
        self.gate_inits = self.encode(train_gate)[0].tolist()
