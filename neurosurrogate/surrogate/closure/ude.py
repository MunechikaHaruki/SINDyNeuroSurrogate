"""UDE の閉包項: 潜在方程式 f を MLP で表す実装。

`sindy/` が「ライブラリ項の疎な線形結合を疎回帰で同定」するのに対し、こちらは
「MLP を ODE 解を通した勾配降下で同定」する。**同定の際 encoder/decoder も同時に
更新される**が、その成果物は preprocessor 側が持つ (kernel の decode 経路が
hybrid と同一で済む) → ここは潜在方程式の重みだけを持つ。

入力は (latent, V)。V はレンジが桁で違う (mV オーダ vs tanh 潜在) ので、
学習時に決めた統計で正規化してから食わせる — その統計も重みと同じく学習成果物。
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from ...core.opcost import OpCost
from .base import Closure

# tanh(x) = 1 - 2 / (exp(2x) + 1) — preprocessor/autoencoder.py と同じ数え方。
TANH_COST = OpCost(exp=1, div=1, pm=2, mul=1)


def mlp(params: list[dict], x: jnp.ndarray) -> jnp.ndarray:
    """tanh MLP。最終層のみ線形。"""
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["W"] + layer["b"])
    return jnp.asarray(x @ params[-1]["W"] + params[-1]["b"])


def latent_deriv(
    layers: list[dict], latent: jnp.ndarray, v_norm: jnp.ndarray, pull: float
) -> jnp.ndarray:
    """潜在の速度 `d(latent)/dt = NN(latent, V) − pull·ReLU(|latent|−1)·sign(latent)`。

    **学習と推論が共有する唯一の右辺** (学習時の rollout と kernel で式が 1 文字でも
    違えば joint 学習の意味が消える)。

    第 2 項は潜在を encoder の値域 [-1,1] へ戻す復元力。encoder は tanh なので学習
    データの潜在は必ずこの中に居る = **範囲外は decoder にとっても NN にとっても未学習**
    で、置換シミュでそこへ出れば任意値を返して発散する。

    復元力は `|latent| > 1` でのみ働き、範囲内では恒等的に 0 = 学習したダイナミクスを
    一切歪めない。**端で速度を 0 にする形 (`·(1-latent²)`) にしてはいけない**: 発散は
    確かに止まるが、代わりに端へ到達した潜在が二度と戻れず凍結する。実測では MC 置換
    時に潜在が 17〜60% の時間 `|z|>0.99` に張り付き (学習データでは 0%)、凍結した
    ゲートが V を壊していた。範囲外に出ること自体は許し、指数的に引き戻す。
    """
    excess = jnp.maximum(jnp.abs(latent) - 1.0, 0.0) * jnp.sign(latent)
    return mlp(layers, jnp.concatenate([latent, v_norm], axis=-1)) - pull * excess


@dataclass
class UDEClosure(Closure):
    """潜在方程式 `f(latent, V) -> d(latent)/dt` を MLP で表す閉包項。

    layers : 各層の {W, b} (np で保持 → そのまま pickle 可能)。
    v_mean / v_std : 学習データの V 統計 (入力正規化。推論時も同じ値を使う)。
    pull : 潜在を値域 [-1,1] へ戻す復元力の強さ (`latent_deriv` 参照)。
    losses : 学習末尾のロス内訳 (MLflow へ流す)。
    """

    layers: list[dict[str, np.ndarray]]
    v_mean: float
    v_std: float
    pull: float
    losses: dict[str, float] = field(default_factory=dict)

    def apply(self) -> Callable:
        """`(latent, V) -> d(latent)/dt` の JAX 関数を組む (kernel が毎ステップ呼ぶ)。

        右辺は学習時の rollout と同じ `latent_deriv` を通す (経路を分けない)。
        """
        layers = [{k: jnp.asarray(v) for k, v in ly.items()} for ly in self.layers]

        def dlatent(latent: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
            v_norm = (jnp.atleast_1d(v) - self.v_mean) / self.v_std
            return latent_deriv(layers, jnp.atleast_1d(latent), v_norm, self.pull)

        return dlatent

    def metrics(self) -> dict[str, float]:
        return {f"ude/{k}": float(v) for k, v in self.losses.items()}

    def opcost(self) -> OpCost:
        # V の正規化 (減算+除算) + 各層の行列積 + 中間層の tanh。
        cost = OpCost(pm=1, div=1)
        for i, layer in enumerate(self.layers):
            n_in, n_out = layer["W"].shape
            cost = cost + OpCost(mul=n_in * n_out, pm=n_in * n_out)
            if i < len(self.layers) - 1:
                cost = cost + TANH_COST * n_out
        # 復元力 |z|-1 → clamp → sign 乗算 → 減算 (潜在 1 本あたり)。
        return cost + OpCost(pm=3, mul=2) * int(self.layers[-1]["W"].shape[1])
