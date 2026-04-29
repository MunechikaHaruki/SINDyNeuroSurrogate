import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# モデル（純粋な関数）
# ------------------------------------------------------------------


def encoder(params, x):
    return jnp.tanh(x @ params["W"] + params["b"])


def decoder(params, z):
    return z @ params["W"] + params["b"]


def loss_fn(params, x):
    z = encoder(params["enc"], x)
    x_hat = decoder(params["dec"], z)
    return jnp.mean((x - x_hat) ** 2)


# ------------------------------------------------------------------
# sklearn互換クラス
# ------------------------------------------------------------------


class AutoEncoderPreprocessor:
    """
    JAX実装のAutoEncoder。PCAと同じfit/transformインターフェース。

    Parameters
    ----------
    n_components : int
        潜在空間の次元数（PCAのn_componentsに相当）。
    epochs : int
        学習エポック数。
    lr : float
        学習率。
    """

    def __init__(self, n_components: int = 1, epochs: int = 200, lr: float = 1e-3):
        self.n_components = n_components
        self.epochs = epochs
        self.lr = lr
        self._params = None
        self._mean = None
        self._std = None

    def _init_params(self, input_dim: int, key: jax.Array) -> dict:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        return {
            "enc": {
                "W": jax.random.normal(k1, (input_dim, self.n_components)) * 0.1,
                "b": jax.random.normal(k2, (self.n_components,)) * 0.1,
            },
            "dec": {
                "W": jax.random.normal(k3, (self.n_components, input_dim)) * 0.1,
                "b": jax.random.normal(k4, (input_dim,)) * 0.1,
            },
        }

    def fit(self, X: np.ndarray) -> "AutoEncoderPreprocessor":
        X = np.asarray(X, dtype=np.float32)

        # 標準化
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_norm = jnp.array((X - self._mean) / self._std)

        # 初期化
        params = self._init_params(X.shape[1], jax.random.PRNGKey(0))
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(params)

        # JITコンパイル
        @jax.jit
        def step(params, opt_state, x):
            loss, grads = jax.value_and_grad(loss_fn)(params, x)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # 学習ループ
        for epoch in range(self.epochs):
            params, opt_state, loss = step(params, opt_state, X_norm)
            if (epoch + 1) % 50 == 0:
                logger.info(
                    f"[AutoEncoder] epoch {epoch + 1}/{self.epochs}  loss={loss:.6f}"
                )

        self._params = params
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("fit()を先に呼んでください。")
        X_norm = jnp.array((np.asarray(X, dtype=np.float32) - self._mean) / self._std)
        return np.array(encoder(self._params["enc"], X_norm))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("fit()を先に呼んでください。")
        x_hat = decoder(self._params["dec"], jnp.array(np.asarray(Z, dtype=np.float32)))
        return np.array(x_hat) * self._std + self._mean
