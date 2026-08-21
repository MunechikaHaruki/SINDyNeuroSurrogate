"""設定ツリー → 学習済み `Surrogate`。

**ドメイン層でここだけが設定の形を知る**。`model.py` (持つ・保存する・適用する) から
組み立てを外に出すことで、設定ブロックの増減が学習済みモデルの型へ届かない。

cfg の 3 ブロックは各構成要素の構築引数そのもので、ここは宛先へ振り分けるだけ
(設定を組み替えない = 構造への暗黙依存を持たない):
  spec         → `SurrogateSpec.from_config` (学習構造 = 実装の dispatch キー)
  preprocessor → 種別固有 hyperparams のみ ┐ この 2 つを学習順に走らせるのは
  ansatz       → 定式化固有 hyperparams のみ ┘ `Ansatz.fit` (前処理 → 閉包項)
"""

from ..core.simulator import unified_simulator
from .model import Surrogate, SurrogateSpec


def fit_surrogate(cfg: dict) -> Surrogate:
    """設定ツリーから学習済み surrogate を組む唯一の入口。"""
    spec = SurrogateSpec.from_config(cfg["spec"])
    ansatz = spec.ansatz()
    # 学習データは spec から決定的に再現できる (`Surrogate.training_data` と同じ式)
    # ので、学習済みモデルへ持ち回さず捨てる。
    preprocessor, closure = ansatz.fit(
        unified_simulator(spec.dataset.materialize()),
        cfg["preprocessor"],
        cfg["ansatz"],
    )
    return Surrogate(spec, ansatz, preprocessor, closure)
