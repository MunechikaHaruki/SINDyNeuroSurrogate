"""fit 済み `Preprocessor` に残る派生量 (初期潜在と再構成統計)。

どちらも encode/decode が確定した後にしか出せず、出し方は全実装で同じ → 契約でなく
前処理側の実装としてここに置く。**セットせず返す**のは、再構成統計が契約の field で
ないから (各実装が自分の `metrics()` のためだけに持つ)。呼ぶのは各 `fit` の末尾と、
encoder/decoder を学習後に差し替える UDE の joint 学習。
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .. import Preprocessor


def fit_artifacts(
    preprocessor: Preprocessor, train_gate: np.ndarray
) -> tuple[list, dict[str, float]]:
    """(学習データ先頭の潜在, 再構成統計) を返す。

    統計のキーは前置きなし — MLflow へ出すときの `pca/` `ae/` は実装が付ける
    (どの前処理の値かはこの層では決まらない)。
    """
    latent = preprocessor.encode(train_gate)
    reconstructed = np.asarray(preprocessor.decode(jnp.asarray(latent)))
    mse = float(np.mean((train_gate - reconstructed) ** 2))
    return latent[0].tolist(), {
        "reconstruction_mse": mse,
        "reconstruction_mse_ratio": mse / float(np.var(train_gate)),
    }
