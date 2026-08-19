"""UDE: encoder / decoder / 潜在方程式を同じロスで同時に学習する hybrid。

既存 hybrid は 2 段 (preprocessor が再構成 MSE で座標を固定 → SINDy が方程式を同定)。
この分離がジレンマを生む: PCA は方程式を張れるが圧縮できず、AE は圧縮できるが潜在が
歪んでレート関数ライブラリが張れない = 圧縮できる座標と方程式が書ける座標が一致しない。
座標と方程式を同じ目的関数で決めれば消える → 潜在方程式も NN にし「潜在を積分した軌道が
元のゲート軌道に合うか」を直接ロスにする。

kernel 骨格・physics 分離・初期値は `HybridAnsatz` が持ち、SINDy 版との差は fit と
潜在方程式の評価 (NN vs ξ 内積) だけ。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr

from ...core import access
from ..closure.ude import UDEClosure, latent_deriv
from ..preprocessor import Preprocessor
from ..preprocessor.autoencoder import AEPreprocessor, decoder, encoder
from .hybrid import HybridAnsatz

if TYPE_CHECKING:
    from ..model import SurrogateSpec

_logger = logging.getLogger(__name__)


def _init_mlp(dims: list[int], key) -> list[dict]:
    """tanh MLP の重み。最終層 0 初期化 = 開始時 d(latent)/dt ≡ 0 (潜在を積分するので
    初期の右辺が暴れると窓終端で発散し勾配が死ぬ)。"""
    layers = []
    for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        key, sub = jax.random.split(key)
        last = i == len(dims) - 2
        scale = 0.0 if last else np.sqrt(1.0 / n_in)
        layers.append(
            {
                "W": jax.random.normal(sub, (n_in, n_out)) * scale,
                "b": jnp.zeros((n_out,)),
            }
        )
    return layers


class UDEAnsatz(HybridAnsatz[UDEClosure]):
    """Hybrid + 潜在方程式を NN にし encoder/decoder ごと ODE 解を通して joint 学習。

    ロス 2 項: traj = 窓頭を encode→Euler rollout→decode した軌道が真のゲート軌道に
    一致するか / recon = 同点の再構成 (decoder の外挿崩れを防ぐアンカー、重み w_recon)。
    kernel 骨格は `HybridAnsatz` から継承し、ここは joint 学習 (fit) と潜在方程式の
    評価 (NN) だけを担う。
    """

    def fit(
        self,
        spec: SurrogateSpec,
        training_data: xr.Dataset,
        preprocessor: Preprocessor,
        config: dict,
    ) -> UDEClosure:
        if not isinstance(preprocessor, AEPreprocessor):
            raise ValueError(
                "ude は encoder/decoder を学習変数として更新するため "
                f"preprocessor_type=ae が要る (指定: {type(preprocessor).__name__})"
            )
        epochs = int(config.get("epochs", 3000))
        lr = float(config.get("lr", 3e-3))
        window = int(config.get("window", 100))
        batch = int(config.get("batch", 128))
        hidden = int(config.get("hidden", 32))
        depth = int(config.get("depth", 2))
        w_recon = float(config.get("w_recon", 1.0))
        # 値域外の復元力。学習も同じ右辺で rollout する (推論だけの後付け項は学習が
        # 知らない力で軌道を曲げる)。
        pull = float(config.get("pull", 20.0))

        comp_ids = spec.train_comp_ids()
        # (n_comp, T, n_gate) / (n_comp, T)。窓は comp を跨がせない (別軌道)。
        training_gates = self.training_gates(spec, training_data)
        gate = jnp.asarray(np.stack(training_gates), dtype=jnp.float32)
        volt = jnp.asarray(
            np.stack(access.potentials(training_data, comp_ids)), dtype=jnp.float32
        )
        n_comp, n_time, _ = gate.shape
        if n_time <= window:
            raise ValueError(f"window={window} が学習軌道長 {n_time} 以上")

        # 標準化統計は preprocessor の fit 済みを流用 (データの素性で表現でない)。V も
        # 同様に固定統計で正規化。
        gate_norm = (gate - jnp.asarray(preprocessor.x_mean)) / jnp.asarray(
            preprocessor.x_std
        )
        v_mean, v_std = float(volt.mean()), float(volt.std() + 1e-8)
        v_norm = (volt - v_mean) / v_std
        dt = float(spec.dataset.dt)

        params: dict[str, Any] = {
            "enc": {k: jnp.asarray(v) for k, v in preprocessor.enc_params.items()},
            "dec": {k: jnp.asarray(v) for k, v in preprocessor.dec_params.items()},
            "nn": _init_mlp(
                [spec.n_components + 1] + [hidden] * depth + [spec.n_components],
                jax.random.PRNGKey(0),
            ),
        }

        def rollout(nn: list[dict], z0: jnp.ndarray, vb: jnp.ndarray) -> jnp.ndarray:
            """窓内を Euler 積分 (simulator と同じ刻み → 学習右辺が置換で回る)。"""

            def step(z, v_k):
                return z + latent_deriv(nn, z, v_k[:, None], pull) * dt, z

            _, zs = jax.lax.scan(step, z0, vb.T)  # (W, B, n_latent)
            return jnp.asarray(zs)

        def loss_fn(params, gb, vb):  # gb (B,W,n_gate) 正規化済 / vb (B,W)
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
            (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, gb, vb
            )
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

        # joint 学習した encoder/decoder を preprocessor へ書き戻す = 前処理が「先に
        # 固定される変換」でなくなる唯一の点 (kernel の decode・gate_inits・再構成指標が
        # 更新後の値になる)。
        preprocessor.enc_params = {k: np.asarray(v) for k, v in params["enc"].items()}
        preprocessor.dec_params = {k: np.asarray(v) for k, v in params["dec"].items()}
        preprocessor._set_fit_artifacts(np.concatenate(training_gates, axis=0))

        return UDEClosure(
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

    def dlatent(
        self,
        spec: SurrogateSpec,
        preprocessor: Preprocessor,
        closure: UDEClosure,
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        return closure.apply()
