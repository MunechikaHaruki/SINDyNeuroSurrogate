from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import xarray as xr

from ....core import access
from ....core.coords import transform_gate
from ....core.network import CompartmentType
from .. import Ansatz, Preprocessor, TrainInputs
from ..closure.sindy import SINDyBundle
from ..closure.sindy.roles import Roles
from ._sindy_fit import fit_sindy

if TYPE_CHECKING:
    from ...model import SurrogateSpec


class SINDyAnsatz(Ansatz[SINDyBundle]):
    def params_match(self, train: tuple | None, node: tuple | None) -> bool:
        """params 完全一致のノードにしか適用できない: dV も含め全部を係数へ同定する
        ので、回路 params が ξ に焼き込まれている。"""
        return train == node

    def n_train_gate(self, spec: SurrogateSpec) -> int:
        """全ゲートを学習 (V+gate を丸ごと同定 → physics へ分離する列が無い)。"""
        return len(spec.comp_type.gate_names)

    def train_inputs(
        self,
        spec: SurrogateSpec,
        training_data: xr.Dataset,
        preprocessor: Preprocessor,
    ) -> TrainInputs:
        # 状態 [V, z1..zN] 丸ごと、入力は流入電流 (transform_gate が I_internal を u
        # 列へ)。comp ごとに 1 軌道 (縦連結は偽微分)。
        comp_ids = spec.train_comp_ids()
        preprocessed = [
            transform_gate(preprocessor, training_data, comp_id=i) for i in comp_ids
        ]
        return TrainInputs(
            x_names=[access.POTENTIAL_VAR, *access.latent_vars(spec.n_components)],
            u_names=["u"],
            x=[
                access.comp_matrix(pre, i)
                for pre, i in zip(preprocessed, comp_ids, strict=True)
            ],
            u=[access.i_ext_values(pre)[:, None] for pre in preprocessed],
        )

    def fit(
        self,
        spec: SurrogateSpec,
        training_data: xr.Dataset,
        preprocessor: Preprocessor,
        config: dict,
    ) -> SINDyBundle:
        inputs = self.train_inputs(spec, training_data, preprocessor)
        n = spec.n_components
        # 列構造: [V, z1..zN, u]。V=0, gate 群, 末尾に外部電流。
        roles = Roles(V=0, g=list(range(1, 1 + n)), u=1 + n)
        return fit_sindy(inputs, access.time(training_data), roles, config)

    def surr_comp_type(
        self,
        spec: SurrogateSpec,
        preprocessor: Preprocessor,
        closure: SINDyBundle,
    ) -> CompartmentType:
        xi = jnp.asarray(closure.xi)
        compute_theta = closure.compute_theta()
        n_latent = spec.n_components

        def surr_kernel(params, i_t, v, state):
            # 束縛順 [V, z1..zN, u]、xi の行も同順 (0=V, 1..=latent)。
            theta = compute_theta(v, *(state[i] for i in range(n_latent)), i_t)
            return xi[0] @ theta, xi[1:] @ theta

        return CompartmentType(
            name=spec.surr_type_name(),
            kernel=surr_kernel,
            param_cls=None,
            gate_names=access.latent_vars(n_latent),
            # param_cls=None → 学習元ノードの初期状態を引き継ぐ (置換は params 完全一致
            # のノードのみ)。
            inits=lambda _: [spec.train_comp().init[0]] + preprocessor.gate_inits,
            opcost=closure.opcost(),  # 丸ごと同定 → コスト = 閉包項の評価
        )
