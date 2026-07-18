"""サロゲート ansatz 群: 元方程式の構造をどれだけ仮定するかで分かれる定式化。

- sindy   : 構造発見 (ライブラリから方程式ごと同定)
- hybrid  : 構造保存 (HH 物理 dV/dt を残しゲート dynamics のみ学習)
- (将来) neuralode : 構造なし (完全データ駆動 RHS)

各 ansatz は NeuroSurrogateBase を継承し、SURR_CLS で type 名解決する。
"""

from pathlib import Path

import joblib

from .base import BUNDLE_FILE, NeuroSurrogateBase, SurrogateMeta
from .hybrid import HybridSINDyNeuroSurrogate
from .sindy import SINDyNeuroSurrogate

SURR_CLS: dict[str, type[NeuroSurrogateBase]] = {
    cls.SURROGATE_TYPE: cls for cls in (SINDyNeuroSurrogate, HybridSINDyNeuroSurrogate)
}


def load_surrogate(dir: Path | str) -> NeuroSurrogateBase:
    data = joblib.load(Path(dir) / BUNDLE_FILE)
    meta: SurrogateMeta = data["meta"]
    cls = SURR_CLS[meta.surrogate_type]
    surrogate = cls.__new__(cls)
    surrogate._meta = meta
    surrogate._sindy_bundle = data["sindy_bundle"]
    surrogate._preprocessor = data["preprocessor"]
    return surrogate
