import jax.numpy as jnp
import numpy as np
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
from .base import Ansatz


class SINDyAnsatz(Ansatz[SINDyBundle]):
    def train_gate(self, meta: SurrogateMeta, train_xr: xr.Dataset) -> np.ndarray:
        return access.gate_matrix(train_xr, meta.train_comp_id)

    def fit(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
        spec: dict,
    ) -> SINDyBundle:
        preprocessed_xr = transform_gate(
            preprocessor, train_xr, comp_id=meta.train_comp_id
        )
        return SINDyBundle.from_sindy(
            library_specs=spec["library_specs"],
            optimizer_spec=spec["optimizer"],
            x=access.comp_matrix(preprocessed_xr, meta.train_comp_id),
            u=access.i_ext_values(preprocessed_xr),
            t=access.time(train_xr),
            targets=[sp.Symbol(v) for v in preprocessed_xr.variable.values],
            inputs=[sp.Symbol("u")],
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
            name="surr",
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
