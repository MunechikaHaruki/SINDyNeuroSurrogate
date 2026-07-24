import jax.numpy as jnp
import xarray as xr

from ....core import access
from ....core.network import CompartmentType
from ...closure.sindy import SINDyBundle
from ...closure.sindy.roles import Roles
from ...meta import SurrogateMeta
from ...preprocessor.base import Preprocessor
from ..base import Ansatz, TrainInputs
from .hybrid_kernel import hybrid_physics, hybrid_surr_comp_type, hybrid_train_inputs
from .sindy_fit import fit_sindy


class HybridAnsatz(Ansatz[SINDyBundle]):
    """Hybrid + 潜在方程式を SINDy で同定 (前処理と別学習、SINDy 入力順 g1..,V)。

    kernel 骨格は `hybrid_kernel.py` の共有関数が持ち、ここは SINDy 固有の同定 (fit) と
    潜在方程式の評価 (ξ 内積) だけを担う。
    """

    def n_train_gate(self, meta: SurrogateMeta) -> int:
        """純電位依存ゲートのみ学習 (Ca サブ系は physics へ分離)。"""
        return hybrid_physics(meta).n_learned

    def train_inputs(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
    ) -> TrainInputs:
        return hybrid_train_inputs(
            self.train_source(meta), train_xr, preprocessor, meta.n_components
        )

    def fit(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
        spec: dict,
    ) -> SINDyBundle:
        inputs = self.train_inputs(meta, train_xr, preprocessor)
        # 列構造: [g1..gN, V]。gate 群が先頭、末尾に V。u は入力に無し。
        roles = Roles(V=meta.n_components, g=list(range(meta.n_components)))
        return fit_sindy(inputs, access.time(train_xr), roles, spec)

    def surr_comp_type(
        self,
        meta: SurrogateMeta,
        preprocessor: Preprocessor,
        closure: SINDyBundle,
    ) -> CompartmentType:
        xi = jnp.asarray(closure.xi)
        compute_theta = closure.compute_theta()
        return hybrid_surr_comp_type(
            meta,
            preprocessor,
            closure,
            lambda latent, v: xi @ compute_theta(*latent, v),
        )
