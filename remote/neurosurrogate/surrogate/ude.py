"""UDE の右辺: 潜在の時間発展を MLP にし、encoder・decoder と同じロスで学習する。

二段 (圧縮器を再構成誤差で固定 → SINDy) だと、圧縮できる座標と式が書ける座標が
一致しない。窓の頭を encode して積分し decode した軌道を、元のゲート軌道に合わせる
ロスで、座標と右辺を一緒に決める。
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..physics.cells import OpCost
from .compress import AE, TANH_COST, decoder, encoder, fill_fit_results

_logger = logging.getLogger(__name__)


def _mlp(layers: list[dict], x: jnp.ndarray) -> jnp.ndarray:
    """tanh の MLP。最終層だけ線形。"""
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["W"] + layer["b"])
    return jnp.asarray(x @ layers[-1]["W"] + layers[-1]["b"])


def latent_deriv(layers, latent, v_norm, pull: float) -> jnp.ndarray:
    """dz/dt = NN(z, V) − pull·ReLU(|z| − 1)·sign(z)。学習と推論で同じ右辺を通す。

    第 2 項は潜在を encoder の値域 [−1, 1] へ戻す復元力で、範囲の内側では 0。端で
    速度を 0 にする形にすると、端に着いた潜在が戻れずに凍る (実測で V を壊した)。
    """
    excess = jnp.maximum(jnp.abs(latent) - 1.0, 0.0) * jnp.sign(latent)
    return _mlp(layers, jnp.concatenate([latent, v_norm], axis=-1)) - pull * excess


@dataclass
class UDE:
    layers: list[dict[str, np.ndarray]]
    v_mean: float  # V の正規化の統計 (学習データから)
    v_std: float
    pull: float
    losses: dict[str, float] = field(default_factory=dict)

    def rhs(self) -> Callable:
        """(z, V) → dz/dt。"""
        layers = [{k: jnp.asarray(v) for k, v in ly.items()} for ly in self.layers]

        def dz(latent, v):
            v_norm = (jnp.atleast_1d(v) - self.v_mean) / self.v_std
            return latent_deriv(layers, jnp.atleast_1d(latent), v_norm, self.pull)

        return dz

    def metrics(self) -> dict[str, float]:
        return {f"ude/{k}": float(v) for k, v in self.losses.items()}

    def describe(self) -> dict[str, Any]:
        return {"kind": "ude", "layers": [list(ly["W"].shape) for ly in self.layers]}

    def opcost(self) -> OpCost:
        """V の正規化、各層の行列積と tanh、復元力。"""
        cost = OpCost(pm=1, div=1)
        for i, layer in enumerate(self.layers):
            n_in, n_out = layer["W"].shape
            cost = cost + OpCost(mul=n_in * n_out, pm=n_in * n_out)
            if i < len(self.layers) - 1:
                cost = cost + TANH_COST * n_out
        return cost + OpCost(pm=3, mul=2) * int(self.layers[-1]["W"].shape[1])


def _init_mlp(dims: list[int], key) -> list[dict]:
    """最終層は 0 で始める (初期の右辺が暴れると窓の終端で発散する)。"""
    layers = []
    for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        key, sub = jax.random.split(key)
        scale = 0.0 if i == len(dims) - 2 else np.sqrt(1.0 / n_in)
        layers.append(
            {
                "W": jax.random.normal(sub, (n_in, n_out)) * scale,
                "b": jnp.zeros((n_out,)),
            }
        )
    return layers


def fit_ude(
    ae: AE,
    gate: jnp.ndarray,
    volt: jnp.ndarray,
    train_gate: np.ndarray,
    *,
    dt: float,
    d: int,
    epochs: int = 3000,
    lr: float = 3e-3,
    window: int = 100,
    batch: int = 128,
    hidden: int = 32,
    depth: int = 2,
    w_recon: float = 1.0,
    pull: float = 20.0,
) -> UDE:
    """ae を更新し、右辺を返す。gate は (区画, 時刻, ゲート)、volt は (区画, 時刻)。"""
    n_comp, n_time, _ = gate.shape
    if n_time <= window:
        raise ValueError(f"window={window} が学習軌道長 {n_time} 以上")
    gate_norm = (gate - jnp.asarray(ae.x_mean)) / jnp.asarray(ae.x_std)
    v_mean, v_std = float(volt.mean()), float(volt.std() + 1e-8)
    v_norm = (volt - v_mean) / v_std

    params: dict[str, Any] = {
        "enc": {k: jnp.asarray(v) for k, v in ae.enc_params.items()},
        "dec": {k: jnp.asarray(v) for k, v in ae.dec_params.items()},
        "nn": _init_mlp([d + 1] + [hidden] * depth + [d], jax.random.PRNGKey(0)),
    }

    def rollout(nn, z0, vb):
        """窓の中を Euler で積分する (シミュレータと同じ刻み)。"""

        def step(z, v_k):
            return z + latent_deriv(nn, z, v_k[:, None], pull) * dt, z

        _, zs = jax.lax.scan(step, z0, vb.T)
        return jnp.asarray(zs)

    def loss_fn(params, gb, vb):
        z0 = encoder(params["enc"], gb[:, 0, :])
        g_hat = decoder(params["dec"], rollout(params["nn"], z0, vb))
        loss_traj = jnp.mean((g_hat - jnp.swapaxes(gb, 0, 1)) ** 2)
        recon = decoder(params["dec"], encoder(params["enc"], gb))
        loss_recon = jnp.mean((recon - gb) ** 2)
        return loss_traj + w_recon * loss_recon, (loss_traj, loss_recon)

    optimizer = optax.adam(lr)

    @jax.jit
    def step(params, opt_state, key):
        c_key, t_key = jax.random.split(key)
        ci = jax.random.randint(c_key, (batch,), 0, n_comp)
        t0 = jax.random.randint(t_key, (batch,), 0, n_time - window)
        idx = t0[:, None] + jnp.arange(window)
        gb, vb = gate_norm[ci[:, None], idx], v_norm[ci[:, None], idx]
        (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, gb, vb)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss, parts

    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(1)
    loss = parts = None
    for epoch in range(epochs):
        key, sub = jax.random.split(key)
        params, opt_state, loss, parts = step(params, opt_state, sub)
        if (epoch + 1) % 200 == 0:
            _logger.info(
                f"[UDE] epoch {epoch + 1}/{epochs}  loss={float(loss):.6f} "
                f"(traj={float(parts[0]):.6f} recon={float(parts[1]):.6f})"
            )
    if loss is None or parts is None:
        raise ValueError(f"epochs={epochs} は 1 以上が要る")

    # 一緒に学習した encoder・decoder を書き戻す (decode・初期値・指標も更新後の値)
    ae.enc_params = {k: np.asarray(v) for k, v in params["enc"].items()}
    ae.dec_params = {k: np.asarray(v) for k, v in params["dec"].items()}
    fill_fit_results(ae, train_gate)
    return UDE(
        layers=[{k: np.asarray(v) for k, v in ly.items()} for ly in params["nn"]],
        v_mean=v_mean,
        v_std=v_std,
        pull=pull,
        losses={
            "loss": float(loss),
            "loss_traj": float(parts[0]),
            "loss_recon": float(parts[1]),
        },
    )
