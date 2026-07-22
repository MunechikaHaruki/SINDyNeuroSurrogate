"""ゲート ↔ 潜在の encode/decode を担う preprocessor 群。

`Preprocessor` 抽象基底 (base.py) に PCA (pca.py) / AE (autoencoder.py) を実装し、
`PREPROCESSOR_CLS` が type 名で解決する (組み立ては bundle.setup が担う)。
"""

from .autoencoder import AEPreprocessor
from .base import Preprocessor
from .pca import PCAPreprocessor

PREPROCESSOR_CLS: dict[str, type[Preprocessor]] = {
    cls.kind: cls for cls in (PCAPreprocessor, AEPreprocessor)
}
