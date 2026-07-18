"""ゲート ↔ 潜在の encode/decode を担う preprocessor 群。

fit 時に学習し、以降 encode/decode/metrics/opcost を **np パラメータ保持で自己完結**
する (生の学習オブジェクトは持たず直列化可能)。Preprocessor を抽象基底に PCA/AE を
実装し、`build` が spec (type + hyperparams) から dispatch する。
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA

from ..core.opcost import OpCost
from .autoencoder import decoder, encoder, train_autoencoder

# tanh(x) = 1 - 2 / (exp(2x) + 1)
TANH_COST = OpCost(exp=1, div=1, pm=2, mul=1)


def _reconstruction_stats(
    encode: Callable, decode: Callable, train_gate: np.ndarray
) -> tuple[float, float]:
    reconstructed = np.asarray(decode(encode(train_gate)))
    mse = float(np.mean((train_gate - reconstructed) ** 2))
    return mse, mse / float(np.var(train_gate))


class Preprocessor(ABC):
    """ゲート ↔ 潜在の可逆変換。学習結果を np で保持し直列化可能。"""

    kind: ClassVar[str]
    # 学習データ先頭の潜在 = 置換シミュの初期ゲート値 (fit で設定)。
    gate_inits: list

    @classmethod
    @abstractmethod
    def fit(cls, train_gate: np.ndarray, spec: dict) -> "Preprocessor":
        """spec (n_components + 種別固有 hyperparams) で学習した preprocessor。"""
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


@dataclass
class PCAPreprocessor(Preprocessor):
    kind: ClassVar[str] = "pca"

    components: np.ndarray
    mean: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    reconstruction_mse: float = 0.0
    reconstruction_mse_ratio: float = 0.0
    gate_inits: list = field(default_factory=list)

    @classmethod
    def fit(cls, train_gate: np.ndarray, spec: dict) -> "PCAPreprocessor":
        pca = PCA(n_components=spec["n_components"]).fit(train_gate)
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


@dataclass
class AEPreprocessor(Preprocessor):
    kind: ClassVar[str] = "ae"

    epochs: int
    lr: float
    enc_params: dict[str, np.ndarray]
    dec_params: dict[str, np.ndarray]
    x_mean: np.ndarray
    x_std: np.ndarray
    reconstruction_mse: float = 0.0
    reconstruction_mse_ratio: float = 0.0
    gate_inits: list = field(default_factory=list)

    @classmethod
    def fit(cls, train_gate: np.ndarray, spec: dict) -> "AEPreprocessor":
        epochs = int(spec.get("epochs", 1000))
        lr = float(spec.get("lr", 3e-2))
        params, mean, std = train_autoencoder(
            train_gate, n_components=spec["n_components"], epochs=epochs, lr=lr
        )
        inst = cls(
            epochs=epochs,
            lr=lr,
            enc_params={k: np.asarray(v) for k, v in params["enc"].items()},
            dec_params={k: np.asarray(v) for k, v in params["dec"].items()},
            x_mean=np.asarray(mean),
            x_std=np.asarray(std),
        )
        inst._set_fit_artifacts(train_gate)
        return inst

    @property
    def n_features(self) -> int:
        return int(self.enc_params["W1"].shape[0])

    def encode(self, x: np.ndarray) -> np.ndarray:
        params = {k: jnp.asarray(v) for k, v in self.enc_params.items()}
        x_norm = (jnp.asarray(np.asarray(x)) - jnp.asarray(self.x_mean)) / jnp.asarray(
            self.x_std
        )
        return np.asarray(encoder(params, x_norm))

    def decode(self, state: jnp.ndarray) -> jnp.ndarray:
        params = {k: jnp.asarray(v) for k, v in self.dec_params.items()}
        x_hat = decoder(params, state)
        return jnp.asarray(x_hat * jnp.asarray(self.x_std) + jnp.asarray(self.x_mean))

    def metrics(self) -> dict:
        return {
            "ae/reconstruction_mse": self.reconstruction_mse,
            "ae/reconstruction_mse_ratio": self.reconstruction_mse_ratio,
        }

    def opcost(self) -> OpCost:
        n_latent, hidden = self.dec_params["W1"].shape
        n_gates = int(self.dec_params["W2"].shape[1])
        return (
            OpCost(mul=n_latent * hidden, pm=n_latent * hidden)  # z @ W1 + b1
            + TANH_COST * int(hidden)
            + OpCost(mul=hidden * n_gates, pm=hidden * n_gates)  # h @ W2 + b2
            + OpCost(mul=n_gates, pm=n_gates)  # 標準化の逆変換 (* std + mean)
        )


PREPROCESSOR_CLS: dict[str, type[Preprocessor]] = {
    cls.kind: cls for cls in (PCAPreprocessor, AEPreprocessor)
}


def build_preprocessor(spec: dict, train_gate: np.ndarray) -> Preprocessor:
    """spec={type, n_components, ...} から preprocessor を学習。"""
    spec = dict(spec)
    return PREPROCESSOR_CLS[spec.pop("type")].fit(train_gate, spec)
