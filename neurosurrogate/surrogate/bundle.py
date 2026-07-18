from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pysindy as ps
import sympy as sp

from ..core.opcost import OpCost

if TYPE_CHECKING:
    from collections.abc import Callable

    from .ansatz.roles import Roles
    from .libraries.entry import FeatureLibrary

OPTIMIZER_CLS: dict[str, type] = {
    "stlsq": ps.optimizers.STLSQ,
}


def _instantiate(spec: dict, registry: dict[str, type]):
    spec = dict(spec)
    return registry[spec.pop("type")](**spec)


@dataclass
class SINDyBundle:
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
        from .libraries.entry import FeatureLibrary

        return FeatureLibrary.build(self.library_specs, self.roles)

    def __getstate__(self) -> dict:
        # feature_library の cache (pickle 不能な lambdify 関数) を落として保存。
        # load 後は library_specs+roles から lazy 再構築される。
        return {k: v for k, v in self.__dict__.items() if k != "feature_library"}

    @classmethod
    def from_sindy(
        cls,
        library_specs: list[dict],
        optimizer_spec: dict,
        x: np.ndarray,
        u: np.ndarray,
        t: np.ndarray,
        targets: list[sp.Symbol],
        inputs: list[sp.Symbol],
        roles: "Roles",
    ) -> "SINDyBundle":
        bundle = cls(
            xi=np.empty(0),
            targets=targets,
            inputs=inputs,
            library_specs=library_specs,
            roles=roles,
        )
        sindy = ps.SINDy(
            feature_library=bundle.feature_library.library,
            optimizer=_instantiate(optimizer_spec, OPTIMIZER_CLS),
        )
        sindy.fit(x, u=u, t=t, feature_names=[str(s) for s in bundle.columns])
        bundle.xi = sindy.coefficients()
        # xi の列は pysindy が並べたもの、feature_exprs は自前展開。両者が同順・同表記
        # であることが opcost/compute_theta/描画すべての前提 → fit 時に照合する。
        if (names := sindy.get_feature_names()) != [
            str(e) for e in bundle.feature_exprs
        ]:
            raise ValueError(f"pysindy の feature 順が展開結果と不一致: {names}")
        return bundle

    def xi_metrics(self) -> dict[str, float]:
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
        from .libraries.entry import op_cost

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
