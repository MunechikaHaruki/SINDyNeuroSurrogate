import jax.numpy as jnp
import sympy as sp
import xarray as xr

from ...core import access
from ...core.coords import transform_gate
from ...core.network import CompartmentType
from ...core.opcost import OpCost
from ..closure.sindy import SINDyBundle
from ..closure.sindy.roles import Roles
from ..meta import SurrogateMeta
from ..preprocessor.base import Preprocessor
from .base import Ansatz, TrainInputs


class SINDyAnsatz(Ansatz[SINDyBundle]):
    def n_train_gate(self, meta: SurrogateMeta) -> int:
        """全ゲートを学習する (V+gate 全体を丸ごと同定する定式化 → physics へ分離
        する列が無い)。"""
        return len(meta.comp_type.gate_names)

    def train_inputs(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
    ) -> TrainInputs:
        # 状態は [V, g1..gN] 丸ごと (V も同定対象)、入力は各ノードへの流入電流
        # (transform_gate が I_internal を u 列へ置く)。comp ごとに 1 軌道で分ける
        # — 縦連結すると境界に偽の時間微分が入る。
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
        return SINDyBundle.from_sindy(
            library_specs=spec["library_specs"],
            optimizer_spec=spec["optimizer"],
            x=inputs.x,
            u=inputs.u,
            t=[access.time(train_xr)] * len(inputs.x),
            targets=[sp.Symbol(v) for v in inputs.x_names],
            inputs=[sp.Symbol(v) for v in inputs.u_names],
            # 列構造: [V, g1..gN, u]。V=0, gate 群, 末尾に外部電流。
            roles=Roles(
                V=0,
                g=list(range(1, 1 + meta.n_components)),
                u=1 + meta.n_components,
            ),
        )

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
            # 列構造 [V, g1..gN, u] の順で束縛。xi の行も同順 (0=V, 1..=latent)。
            theta = compute_theta(v, *(state[i] for i in range(n_latent)), i_t)
            return xi[0] @ theta, xi[1:] @ theta

        return CompartmentType(
            name=meta.surr_type_name,
            kernel=surr_kernel,
            param_cls=None,
            gate_names=access.latent_vars(n_latent),
            # surr は params を持たない (param_cls=None) → 学習元ノードの初期状態を
            # そのまま引き継ぐ。置換は params 完全一致のノードにしか起きない。
            inits=lambda _: [meta.train_comp.init[0]] + preprocessor.gate_inits,
            opcost=None,
        )

    def opcost(
        self,
        meta: SurrogateMeta,
        preprocessor: Preprocessor,
        closure: SINDyBundle,
    ) -> OpCost:
        return closure.opcost()
