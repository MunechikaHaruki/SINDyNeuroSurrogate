"""群のゲート ↔ 潜在の圧縮器 (PCA と AutoEncoder)。

どちらも encode・decode・指標・中身・decode の演算コストを持つ。fit の末尾で、学習
データの先頭の潜在 (置換シミュの初期値 gate_inits) と再構成の統計を埋める。
"""

import logging
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.decomposition import PCA as _SklearnPCA

from ..physics.cells import OpCost

_logger = logging.getLogger(__name__)

# tanh(x) = 1 - 2 / (exp(2x) + 1)
TANH_COST = OpCost(exp=1, div=1, pm=2, mul=1)


def fill_fit_results(compressor: Any, train_gate: np.ndarray) -> None:
    """学習データ先頭の潜在と、再構成の統計を埋める。"""
    latent = compressor.encode(train_gate)
    reconstructed = np.asarray(compressor.decode(jnp.asarray(latent)))
    mse = float(np.mean((train_gate - reconstructed) ** 2))
    compressor.gate_inits = latent[0].tolist()
    compressor.reconstruction = {
        "reconstruction_mse": mse,
        "reconstruction_mse_ratio": mse / float(np.var(train_gate)),
    }


# --- PCA -------------------------------------------------------------------------


class PCA:
    gate_inits: list[float]
    reconstruction: dict[str, float]

    def __init__(self, components, mean, explained_variance_ratio, full_ratio):
        self.components = components
        self.mean = mean
        self.explained_variance_ratio = explained_variance_ratio
        # 捨てた成分も含む寄与率。d が累積の飽和点に対し妥当かを見る
        self.full_explained_variance_ratio = full_ratio

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.asarray((np.asarray(x) - self.mean) @ self.components.T)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return z @ jnp.asarray(self.components) + jnp.asarray(self.mean)

    def metrics(self) -> dict[str, float]:
        ratios = {}
        for i, r in enumerate(self.explained_variance_ratio):
            ratios[f"pca/explained_variance_ratio_{i + 1}"] = float(r)
        return {
            **ratios,
            "pca/cumulative_explained_variance_ratio": float(
                self.explained_variance_ratio.sum()
            ),
            **{f"pca/{k}": v for k, v in self.reconstruction.items()},
        }

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "pca",
            "explained_variance_ratio": self.full_explained_variance_ratio.tolist(),
            "n_components": len(self.explained_variance_ratio),
        }

    def opcost(self) -> OpCost:
        """decode: ゲートごとに潜在数の積和と mean の加算。"""
        n_latent, n_gates = self.components.shape
        return OpCost(mul=n_latent * n_gates, pm=n_latent * n_gates)


def fit_pca(gates: np.ndarray, d: int) -> PCA:
    """全成分で 1 度 fit し、上位 d 本を採る。"""
    pca = _SklearnPCA().fit(gates)
    inst = PCA(
        components=np.asarray(pca.components_[:d]),
        mean=np.asarray(pca.mean_),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_[:d]),
        full_ratio=np.asarray(pca.explained_variance_ratio_),
    )
    fill_fit_results(inst, gates)
    return inst


# --- AutoEncoder ------------------------------------------------------------------


def encoder(params, x):
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    return jnp.tanh(h @ params["W2"] + params["b2"])


def decoder(params, z):
    h = jnp.tanh(z @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


class AE:
    gate_inits: list[float]
    reconstruction: dict[str, float]

    def __init__(self, enc_params, dec_params, x_mean, x_std):
        self.enc_params = enc_params
        self.dec_params = dec_params
        self.x_mean = x_mean
        self.x_std = x_std

    def encode(self, x: np.ndarray) -> np.ndarray:
        params = {k: jnp.asarray(v) for k, v in self.enc_params.items()}
        x_norm = (jnp.asarray(np.asarray(x)) - jnp.asarray(self.x_mean)) / jnp.asarray(
            self.x_std
        )
        return np.asarray(encoder(params, x_norm))

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        params = {k: jnp.asarray(v) for k, v in self.dec_params.items()}
        x_hat = decoder(params, z)
        return jnp.asarray(x_hat * jnp.asarray(self.x_std) + jnp.asarray(self.x_mean))

    def metrics(self) -> dict[str, float]:
        return {f"ae/{k}": v for k, v in self.reconstruction.items()}

    def describe(self) -> dict[str, Any]:
        return {"kind": "ae", "hidden": int(self.dec_params["W1"].shape[1])}

    def opcost(self) -> OpCost:
        n_latent, hidden = self.dec_params["W1"].shape
        n_gates = int(self.dec_params["W2"].shape[1])
        return (
            OpCost(mul=n_latent * hidden, pm=n_latent * hidden)  # z @ W1 + b1
            + TANH_COST * int(hidden)
            + OpCost(mul=hidden * n_gates, pm=hidden * n_gates)  # h @ W2 + b2
            + OpCost(mul=n_gates, pm=n_gates)  # 標準化の逆変換
        )


def _ae_init(input_dim: int, d: int, key) -> dict:
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
    hidden = 16
    return {
        "enc": {
            "W1": jax.random.normal(k1, (input_dim, hidden)) * 0.1,
            "b1": jax.random.normal(k2, (hidden,)) * 0.1,
            "W2": jax.random.normal(k3, (hidden, d)) * 0.1,
            "b2": jax.random.normal(k4, (d,)) * 0.1,
        },
        "dec": {
            "W1": jax.random.normal(k5, (d, hidden)) * 0.1,
            "b1": jax.random.normal(k6, (hidden,)) * 0.1,
            "W2": jax.random.normal(k7, (hidden, input_dim)) * 0.1,
            "b2": jax.random.normal(k8, (input_dim,)) * 0.1,
        },
    }


def _ae_loss(params, x):
    x_hat = decoder(params["dec"], encoder(params["enc"], x))
    return jnp.mean((x - x_hat) ** 2)


def fit_ae(gates: np.ndarray, d: int, *, epochs: int = 1000, lr: float = 3e-2) -> AE:
    """標準化したゲートで、再構成誤差を Adam で下げる。乱数種は固定。"""
    x = np.asarray(gates, dtype=np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    x_norm = jnp.array((x - mean) / std)

    params = _ae_init(x.shape[1], d, jax.random.PRNGKey(0))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x):
        loss, grads = jax.value_and_grad(_ae_loss)(params, x)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    for epoch in range(epochs):
        params, opt_state, loss = step(params, opt_state, x_norm)
        if (epoch + 1) % 50 == 0:
            _logger.info(f"[AutoEncoder] epoch {epoch + 1}/{epochs}  loss={loss:.6f}")
    inst = AE(
        enc_params={k: np.asarray(v) for k, v in params["enc"].items()},
        dec_params={k: np.asarray(v) for k, v in params["dec"].items()},
        x_mean=np.asarray(mean),
        x_std=np.asarray(std),
    )
    fill_fit_results(inst, gates)
    return inst


FITS: dict[str, Callable[..., Any]] = {"pca": fit_pca, "ae": fit_ae}
