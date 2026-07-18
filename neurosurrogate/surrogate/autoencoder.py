import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# モデル（純粋な関数）
# ------------------------------------------------------------------


# 1. encoder関数
def encoder(params, x):
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    return jnp.tanh(h @ params["W2"] + params["b2"])


# 2. decoder関数
def decoder(params, z):
    h = jnp.tanh(z @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


# 3. _init_params（クラス内）


def loss_fn(params, x):
    z = encoder(params["enc"], x)
    x_hat = decoder(params["dec"], z)
    return jnp.mean((x - x_hat) ** 2)


# ------------------------------------------------------------------
# 学習 (パラメータ → 直列化は AEPreprocessor が担う)
# ------------------------------------------------------------------


def _init_params(input_dim: int, n_components: int, key) -> dict:
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
    hidden = 16
    return {
        "enc": {
            "W1": jax.random.normal(k1, (input_dim, hidden)) * 0.1,
            "b1": jax.random.normal(k2, (hidden,)) * 0.1,
            "W2": jax.random.normal(k3, (hidden, n_components)) * 0.1,
            "b2": jax.random.normal(k4, (n_components,)) * 0.1,
        },
        "dec": {
            "W1": jax.random.normal(k5, (n_components, hidden)) * 0.1,
            "b1": jax.random.normal(k6, (hidden,)) * 0.1,
            "W2": jax.random.normal(k7, (hidden, input_dim)) * 0.1,
            "b2": jax.random.normal(k8, (input_dim,)) * 0.1,
        },
    }


def train_autoencoder(
    X: np.ndarray, n_components: int, epochs: int, lr: float
) -> tuple[dict, np.ndarray, np.ndarray]:
    """AutoEncoder を学習し (params, x_mean, x_std) を返す。標準化込み。"""
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = jnp.array((X - mean) / std)

    params = _init_params(X.shape[1], n_components, jax.random.PRNGKey(0))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x):
        loss, grads = jax.value_and_grad(loss_fn)(params, x)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    for epoch in range(epochs):
        params, opt_state, loss = step(params, opt_state, X_norm)
        if (epoch + 1) % 50 == 0:
            logger.info(f"[AutoEncoder] epoch {epoch + 1}/{epochs}  loss={loss:.6f}")
    return params, mean, std
