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

    def __init__(self, n_components: int = 1, epochs: int = 1000, lr: float = 3e-2):
        self.n_components = n_components
        self.epochs = epochs
        self.lr = lr
        self._params: dict | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _init_params(self, input_dim, key):
        k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
        hidden = 16
        return {
            "enc": {
                "W1": jax.random.normal(k1, (input_dim, hidden)) * 0.1,
                "b1": jax.random.normal(k2, (hidden,)) * 0.1,
                "W2": jax.random.normal(k3, (hidden, self.n_components)) * 0.1,
                "b2": jax.random.normal(k4, (self.n_components,)) * 0.1,
            },
            "dec": {
                "W1": jax.random.normal(k5, (self.n_components, hidden)) * 0.1,
                "b1": jax.random.normal(k6, (hidden,)) * 0.1,
                "W2": jax.random.normal(k7, (hidden, input_dim)) * 0.1,
                "b2": jax.random.normal(k8, (input_dim,)) * 0.1,
            },
        }

    @property
    def n_features_in_(self) -> int:
        """学習入力ゲート数 (PCA.n_features_in_ 互換。transform_gate の幅整合用)。"""
        assert self._mean is not None
        return int(self._mean.shape[0])

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
        if self._params is None or self._mean is None or self._std is None:
            raise RuntimeError("fit()を先に呼んでください。")
        X_norm = jnp.array((np.asarray(X, dtype=np.float32) - self._mean) / self._std)
        return np.array(encoder(self._params["enc"], X_norm))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self._params is None or self._mean is None or self._std is None:
            raise RuntimeError("fit()を先に呼んでください。")
        x_hat = decoder(self._params["dec"], jnp.array(np.asarray(Z, dtype=np.float32)))
        return np.array(x_hat) * self._std + self._mean  # type: ignore[no-any-return]
