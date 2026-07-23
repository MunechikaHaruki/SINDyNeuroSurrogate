"""UDE: encoder / decoder / 潜在方程式を **同じロスで同時に** 学習する hybrid。

既存の hybrid は 2 段階だった: preprocessor が再構成 MSE だけを見て潜在座標を決め、
その座標を固定したまま SINDy が潜在方程式を同定する。この分離が次元削減の頭を
押さえている:

  - PCA は線形なので真の `dg/dt = α(1-g) - βg` が潜在でもそのまま張れるが、
    線形部分空間ではゲート多様体を圧縮できない。
  - AE は多様体に沿って圧縮できるが、潜在座標が非線形に歪むのでレート関数
    ライブラリ (catalog の `*_gate_forward` 等) が張れなくなる。

**圧縮できる座標と方程式が書ける座標が一致しない**、というのがジレンマの中身。
座標と方程式を同じ目的関数で決めれば消える → 潜在方程式も NN にして
「潜在を積分した軌道が元のゲート軌道に合うか」を直接ロスにする。

`HybridBase` との差分は `fit` / `_dlatent` / `_closure_opcost` の 3 本だけで、
kernel の骨格・physics 分離 (Ca サブ系)・初期値は hybrid と完全に共有する。
"""

import logging
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr

from ...core import access
from ...core.opcost import OpCost
from ..closure.ude import UDEClosure, latent_deriv
from ..meta import SurrogateMeta
from ..preprocessor.autoencoder import AEPreprocessor, decoder, encoder
from ..preprocessor.base import Preprocessor
from .hybrid import HybridBase

logger = logging.getLogger(__name__)


def _init_mlp(dims: list[int], key) -> list[dict]:
    """tanh MLP の重み。最終層は 0 初期化 = 学習開始時 d(latent)/dt ≡ 0。

    潜在を積分して回すので、初期の右辺が暴れると窓の終端で発散して勾配が死ぬ。
    ゼロから立ち上げれば最初の数 epoch は必ず有限に留まる。
    """
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


class UDEAnsatz(HybridBase[UDEClosure]):
    """潜在方程式を NN にし、encoder/decoder ごと ODE 解を通して joint 学習する。

    ロスは 2 項:
      traj  … 窓の先頭を encode → 潜在を Euler で rollout → decode した**軌道**が
              真のゲート軌道に一致するか。学習と評価がどちらも閉ループになる
              (従来は真の点での 1 ステップ誤差しか見ていなかった)。
      recon … 同点の再構成。decoder が潜在の外挿で崩れないためのアンカーで、
              これが従来 preprocessor の目的関数そのもの (重み `w_recon` で従来との
              間を連続に振れる)。
    """

    def fit(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
        spec: dict,
    ) -> UDEClosure:
        if not isinstance(preprocessor, AEPreprocessor):
            raise ValueError(
                "ude は encoder/decoder を学習変数として更新するため "
                f"preprocessor_type=ae が要る (指定: {type(preprocessor).__name__})"
            )
        epochs = int(spec.get("epochs", 3000))
        lr = float(spec.get("lr", 3e-3))
        window = int(spec.get("window", 100))
        batch = int(spec.get("batch", 128))
        hidden = int(spec.get("hidden", 32))
        depth = int(spec.get("depth", 2))
        w_recon = float(spec.get("w_recon", 1.0))
        # 潜在が値域外へ出たときの復元力。学習中も同じ右辺で rollout する (推論だけに
        # 効く項を後付けすると、学習が知らない力で軌道が曲がる)。
        pull = float(spec.get("pull", 20.0))

        source = self.train_source(meta)
        # (n_comp, T, n_gate) / (n_comp, T)。窓は comp を跨がせない (別の軌道なので
        # 連結すると境界に嘘の遷移が入る) → comp 軸を残したまま窓を切る。
        gate = jnp.asarray(
            np.stack([source.gate(train_xr, i) for i in source.comp_ids]),
            dtype=jnp.float32,
        )
        volt = jnp.asarray(
            np.stack([access.potential(train_xr, i) for i in source.comp_ids]),
            dtype=jnp.float32,
        )
        n_comp, n_time, _ = gate.shape
        if n_time <= window:
            raise ValueError(f"window={window} が学習軌道長 {n_time} 以上")

        # 標準化統計は preprocessor の fit 済みのものを流用し、学習変数にはしない
        # (データの素性であって表現ではない)。V も同じ理由で固定統計で正規化する。
        gate_norm = (gate - jnp.asarray(preprocessor.x_mean)) / jnp.asarray(
            preprocessor.x_std
        )
        v_mean, v_std = float(volt.mean()), float(volt.std() + 1e-8)
        v_norm = (volt - v_mean) / v_std
        dt = float(meta.dataset.dt)

        params: dict[str, Any] = {
            "enc": {k: jnp.asarray(v) for k, v in preprocessor.enc_params.items()},
            "dec": {k: jnp.asarray(v) for k, v in preprocessor.dec_params.items()},
            "nn": _init_mlp(
                [meta.n_components + 1] + [hidden] * depth + [meta.n_components],
                jax.random.PRNGKey(0),
            ),
        }

        def rollout(nn: list[dict], z0: jnp.ndarray, vb: jnp.ndarray) -> jnp.ndarray:
            """窓内を Euler で積分。simulator の generic_euler_solver と同じ刻み方
            (学習した右辺がそのまま置換シミュで回る)。"""

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
                logger.info(
                    f"[UDE] epoch {epoch + 1}/{epochs}  loss={float(loss):.6f} "
                    f"(traj={float(parts[0]):.6f} recon={float(parts[1]):.6f})"
                )
        if loss is None or parts is None:
            raise ValueError(f"epochs={epochs} は 1 以上が要る")

        # joint 学習した encoder/decoder を preprocessor へ書き戻す。**これが UDE で
        # 前処理が「先に固定される変換」でなくなる唯一の点** — kernel が呼ぶ decode
        # も、初期潜在 (gate_inits) も、再構成指標も、ここで更新したものが使われる。
        preprocessor.enc_params = {k: np.asarray(v) for k, v in params["enc"].items()}
        preprocessor.dec_params = {k: np.asarray(v) for k, v in params["dec"].items()}
        preprocessor._set_fit_artifacts(np.asarray(source.stacked_gate(train_xr)))

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

    def _dlatent(self, closure: UDEClosure) -> Callable:
        return closure.apply()

    def _closure_opcost(self, closure: UDEClosure) -> OpCost:
        return closure.opcost()
