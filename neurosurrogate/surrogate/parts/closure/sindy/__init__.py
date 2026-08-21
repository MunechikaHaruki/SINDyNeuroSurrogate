"""SINDy 同定層。

同定結果 (`SINDyBundle`: xi/feature 展開/compute_theta/opcost) をここに、項ライブラリ
を entry.py (ロジック) / _catalog.py (データ) に置く。ansatz/ は方程式の列構造 (roles)
を決め、この層は「その列構造で何を同定するか」だけを担う。

入口 `from_sindy` は ansatz が組んだ `TrainInputs` をそのまま受ける (定義は
`parts/train_inputs.py` にあり、`Closure` 契約と同じく上から引ける) — 素データへ開く
変換を ansatz 側の adapter に置くと、列名と軌道の対応が両者に分かれて崩せるようになる。
"""

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pysindy as ps
import sympy as sp

from .....core.opcost import OpCost
from ... import Closure
from ...train_inputs import TrainInputs

if TYPE_CHECKING:
    from collections.abc import Callable

    from .entry import FeatureLibrary
    from .roles import Roles

_OPTIMIZER_CLS: dict[str, type] = {
    "stlsq": ps.optimizers.STLSQ,
    "sr3": ps.optimizers.SR3,
}
# 疎回帰の既定。**yaml でなくここが単一源** (preset が黙って継承する場所に置くと、
# 別の定式化へも流れる)。preset が optimizer を書けば丸ごとこれに代わる。
_DEFAULT_OPTIMIZER = {
    "type": "stlsq",
    "threshold": 0.01,
    "normalize_columns": False,
    "alpha": 2.0,
}


def _instantiate(spec: dict, registry: dict[str, type]):
    # None は「その optimizer では使わない hyperparam」= yaml で明示 null にして落とす。
    spec = {k: v for k, v in spec.items() if v is not None}
    return registry[spec.pop("type")](**spec)


@dataclass
class SINDyBundle(Closure):
    """閉包項を「ライブラリ項の疎な線形結合」で表す実装 (ξ を疎回帰で同定)。"""

    xi: np.ndarray
    targets: list[sp.Symbol]
    inputs: list[sp.Symbol]
    library_specs: list[dict]
    roles: "Roles"

    @property
    def columns(self) -> list[sp.Symbol]:
        """SINDy 入力行列の列シンボル (roles の列 index が指す先)。"""
        return self.targets + self.inputs

    @property
    def feature_exprs(self) -> list[sp.Expr]:
        """xi の列に対応する feature 式 (列順 = compute_theta = pysindy の feature
        順。from_sindy が pysindy 名との一致を検証済み)。feature の同一性はこの式で
        表され、コスト (op_cost) も表示 (str/latex) もここから派生する。"""
        return self.feature_library.bound_exprs(self.columns)

    @cached_property
    def feature_library(self) -> "FeatureLibrary":
        """役割束縛済み FeatureLibrary (compute_theta/opcost 共用)。lambdify 関数は
        pickle 不能 → field でなく cache 化し、__getstate__ で保存対象から除外する。"""
        from .entry import FeatureLibrary

        return FeatureLibrary.build(self.library_specs, self.roles)

    def __getstate__(self) -> dict:
        # feature_library の cache (pickle 不能な lambdify 関数) を落として保存。
        # load 後は library_specs+roles から lazy 再構築される。
        return {k: v for k, v in self.__dict__.items() if k != "feature_library"}

    @classmethod
    def from_sindy(
        cls,
        inputs: TrainInputs,
        t: np.ndarray,
        roles: "Roles",
        *,
        library_specs: list[dict],
        optimizer: dict = _DEFAULT_OPTIMIZER,
    ) -> "SINDyBundle":
        """同定器へ渡す入力一式から ξ を疎回帰で同定する。

        `TrainInputs` をそのまま受ける: 列名 → Symbol も、comp ごとの軌道を pysindy の
        複数軌道形式へ並べるのも**この入口の仕事**。列構造そのもの (roles) だけは
        定式化ごとに違うので ansatz が組んで渡す。

        hyperparams は ansatz が `**spec.closure_config` で展開して渡す = 既定値は
        この署名 1 箇所、綴り違いは黙って既定へ落ちずに TypeError になる。
        """
        bundle = cls(
            xi=np.empty(0),
            targets=inputs.target_symbols(),
            inputs=inputs.input_symbols(),
            library_specs=library_specs,
            roles=roles,
        )
        sindy = ps.SINDy(
            feature_library=bundle.feature_library.library,
            optimizer=_instantiate(optimizer, _OPTIMIZER_CLS),
        )
        # list で渡すと pysindy が複数軌道として扱う (軌道跨ぎの微分を取らない)。
        sindy.fit(
            inputs.x,
            u=inputs.u,
            t=[t] * len(inputs.x),
            feature_names=[str(s) for s in bundle.columns],
        )
        bundle.xi = sindy.coefficients()
        # xi の列は pysindy が並べたもの、feature_exprs は自前展開。両者が同順・同表記
        # であることが opcost/compute_theta/描画すべての前提 → fit 時に照合する。
        if (names := sindy.get_feature_names()) != [
            str(e) for e in bundle.feature_exprs
        ]:
            raise ValueError(f"pysindy の feature 順が展開結果と不一致: {names}")
        return bundle

    def metrics(self) -> dict[str, float]:
        nnz = int((self.xi != 0).sum())
        return {"nnz": nnz, "nnz_ratio": nnz / self.xi.size}

    def compute_theta(self) -> "Callable":
        subs = self.feature_library.sub_libraries

        def compute_theta(*inputs):
            values = []
            for sub in subs:
                bound = [inputs[i] for i in sub.inputs]
                for entry in sub.entries:
                    values.append(entry.func(*bound))
            return jnp.array(values, dtype=jnp.float64)

        return compute_theta

    def opcost(self) -> OpCost:
        """ξ の積和コスト + 生き残った feature 式の評価コスト (式木から直接算出)。"""
        from .entry import op_cost

        nnz = np.count_nonzero(self.xi).item()
        return sum(
            (
                op_cost(expr)
                for expr, active in zip(
                    self.feature_exprs, np.any(self.xi != 0, axis=0), strict=True
                )
                if active
            ),
            OpCost(mul=nnz, pm=max(0, nnz - int(self.xi.shape[0]))),
        )
