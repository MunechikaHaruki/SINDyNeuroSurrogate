import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ....core.opcost import OpCost
from .. import Preprocessor
from .fit_artifacts import fit_artifacts

_logger = logging.getLogger(__name__)

# tanh(x) = 1 - 2 / (exp(2x) + 1)
_TANH_COST = OpCost(exp=1, div=1, pm=2, mul=1)


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


def _loss_fn(params, x):
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


def _train_autoencoder(
    X: np.ndarray, n_components: int, *, epochs: int = 1000, lr: float = 3e-2
) -> tuple[dict, np.ndarray, np.ndarray]:
    """AutoEncoder を学習し (params, x_mean, x_std) を返す。標準化込み。

    hyperparams の**既定値はこの署名 1 箇所** (`fit` が config を展開して渡す)。
    """
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = jnp.array((X - mean) / std)

    params = _init_params(X.shape[1], n_components, jax.random.PRNGKey(0))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x):
        loss, grads = jax.value_and_grad(_loss_fn)(params, x)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    for epoch in range(epochs):
        params, opt_state, loss = step(params, opt_state, X_norm)
        if (epoch + 1) % 50 == 0:
            _logger.info(f"[AutoEncoder] epoch {epoch + 1}/{epochs}  loss={loss:.6f}")
    return params, mean, std


class AEPreprocessor(Preprocessor):
    # gate_inits と並んで fit 末尾で埋まる (UDE の joint 学習後は上書きされる)。
    # 契約に載らない (metrics でしか読まない) ので実装側で宣言する。
    reconstruction: dict[str, float]

    def __init__(
        self,
        enc_params: dict[str, np.ndarray],
        dec_params: dict[str, np.ndarray],
        x_mean: np.ndarray,
        x_std: np.ndarray,
    ):
        # hyperparams (epochs/lr) は保持しない — 由来は spec.json の
        # preprocessor_config が持ち、学習後の振る舞いには効かない。
        self.enc_params = enc_params
        self.dec_params = dec_params
        self.x_mean = x_mean
        self.x_std = x_std

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

    def metrics(self) -> dict[str, float]:
        return {f"ae/{k}": v for k, v in self.reconstruction.items()}

    def opcost(self) -> OpCost:
        n_latent, hidden = self.dec_params["W1"].shape
        n_gates = int(self.dec_params["W2"].shape[1])
        return (
            OpCost(mul=n_latent * hidden, pm=n_latent * hidden)  # z @ W1 + b1
            + _TANH_COST * int(hidden)
            + OpCost(mul=hidden * n_gates, pm=hidden * n_gates)  # h @ W2 + b2
            + OpCost(mul=n_gates, pm=n_gates)  # 標準化の逆変換 (* std + mean)
        )


def fit_ae(
    train_gates: list[np.ndarray],
    n_components: int,
    *,
    epochs: int = 1000,
    lr: float = 3e-2,
) -> AEPreprocessor:
    """comp ごとの学習ゲート軌道から AutoEncoder を学習する (**この層の入口**)。

    軌道を跨いで縦連結するのは `fit_pca` と同じ理由 (座標変換にとっては点の集合)。
    hyperparams の**既定値はこの署名 1 箇所**で、綴り違いは黙って既定へ落ちずに
    TypeError になる。
    """
    train_gate = np.concatenate(train_gates, axis=0)
    params, mean, std = _train_autoencoder(
        train_gate, n_components, epochs=epochs, lr=lr
    )
    inst = AEPreprocessor(
        enc_params={k: np.asarray(v) for k, v in params["enc"].items()},
        dec_params={k: np.asarray(v) for k, v in params["dec"].items()},
        x_mean=np.asarray(mean),
        x_std=np.asarray(std),
    )
    inst.gate_inits, inst.reconstruction = fit_artifacts(inst, train_gate)
    return inst
