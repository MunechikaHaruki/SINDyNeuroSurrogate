"""ゲート ↔ 潜在の encode/decode を担う preprocessor 群。

`Preprocessor` 抽象基底 (base.py) に PCA (pca.py) / AE (autoencoder.py) を実装し、
`build_preprocessor` が spec (type + hyperparams) から dispatch する。
"""

import numpy as np

from .autoencoder import AEPreprocessor
from .base import Preprocessor
from .pca import PCAPreprocessor

PREPROCESSOR_CLS: dict[str, type[Preprocessor]] = {
    cls.kind: cls for cls in (PCAPreprocessor, AEPreprocessor)
}


def build_preprocessor(spec: dict, train_gate: np.ndarray) -> Preprocessor:
    """spec={type, n_components, ...} から preprocessor を学習。"""
    spec = dict(spec)
    return PREPROCESSOR_CLS[spec.pop("type")].fit(train_gate, spec)
