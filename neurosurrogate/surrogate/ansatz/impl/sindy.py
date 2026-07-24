import jax.numpy as jnp
import xarray as xr

from ....core import access
from ....core.coords import transform_gate
from ....core.network import CompartmentType
from ...closure.sindy import SINDyBundle
from ...closure.sindy.roles import Roles
from ...meta import SurrogateMeta
from ...preprocessor.base import Preprocessor
from ..base import Ansatz, TrainInputs
from .sindy_fit import fit_sindy


class SINDyAnsatz(Ansatz[SINDyBundle]):
    def n_train_gate(self, meta: SurrogateMeta) -> int:
        """全ゲートを学習 (V+gate を丸ごと同定 → physics へ分離する列が無い)。"""
        return len(meta.comp_type.gate_names)

    def train_inputs(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
    ) -> TrainInputs:
        # 状態 [V, g1..gN] 丸ごと、入力は流入電流 (transform_gate が I_internal を u
        # 列へ)。comp ごとに 1 軌道 (縦連結は偽微分)。
        comp_ids = self.train_source(meta).comp_ids
        preprocessed = [
            transform_gate(preprocessor, train_xr, comp_id=i) for i in comp_ids
        ]
        return TrainInputs(
            x_names=[access.POTENTIAL_VAR, *access.latent_vars(meta.n_components)],
            u_names=["u"],
            x=[
                access.comp_matrix(pre, i)
                for pre, i in zip(preprocessed, comp_ids, strict=True)
            ],
            u=[access.i_ext_values(pre)[:, None] for pre in preprocessed],
        )

    def fit(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
        spec: dict,
    ) -> SINDyBundle:
        inputs = self.train_inputs(meta, train_xr, preprocessor)
        n = meta.n_components
        # 列構造: [V, g1..gN, u]。V=0, gate 群, 末尾に外部電流。
        roles = Roles(V=0, g=list(range(1, 1 + n)), u=1 + n)
        return fit_sindy(inputs, access.time(train_xr), roles, spec)

    def surr_comp_type(
        self,
        meta: SurrogateMeta,
        preprocessor: Preprocessor,
        closure: SINDyBundle,
    ) -> CompartmentType:
        xi = jnp.asarray(closure.xi)
        compute_theta = closure.compute_theta()
        n_latent = meta.n_components

        def surr_kernel(params, i_t, v, state):
            # 束縛順 [V, g1..gN, u]、xi の行も同順 (0=V, 1..=latent)。
            theta = compute_theta(v, *(state[i] for i in range(n_latent)), i_t)
            return xi[0] @ theta, xi[1:] @ theta

        return CompartmentType(
            name=meta.surr_type_name,
            kernel=surr_kernel,
            param_cls=None,
            gate_names=access.latent_vars(n_latent),
            # param_cls=None → 学習元ノードの初期状態を引き継ぐ (置換は params 完全一致
            # のノードのみ)。
            inits=lambda _: [meta.train_comp.init[0]] + preprocessor.gate_inits,
            opcost=closure.opcost(),  # 丸ごと同定 → コスト = 閉包項の評価
        )
