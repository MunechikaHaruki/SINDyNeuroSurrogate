"""サロゲート ansatz 群: 元方程式の構造をどれだけ仮定するかで分かれる定式化。

- sindy   : 構造発見 (ライブラリから方程式ごと同定)
- hybrid  : 構造保存 (HH 物理 dV/dt を残しゲート dynamics のみ学習)
- (将来) neuralode : 構造なし (完全データ駆動 RHS)

各 ansatz は Ansatz を継承した状態なしストラテジで、SurrogateBundle が
SURR_CLS で type 名解決して使う。
"""

from .base import Ansatz
from .hybrid import HybridAnsatz
from .sindy import SINDyAnsatz

SURR_CLS: dict[str, type[Ansatz]] = {
    cls.SURROGATE_TYPE: cls for cls in (SINDyAnsatz, HybridAnsatz)
}
