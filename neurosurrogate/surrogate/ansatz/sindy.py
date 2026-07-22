from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import sympy as sp

from ...core import access
from ...core.coords import transform_gate
from ...core.network import CompartmentType
from ...core.opcost import OpCost
from ..sindy import SINDyBundle
from .base import Ansatz
from .roles import Roles

if TYPE_CHECKING:
    from ..bundle import SurrogateBundle


class SINDyAnsatz(Ansatz):
    SURROGATE_TYPE = "sindy"

    def train_gate(self, bundle: "SurrogateBundle") -> np.ndarray:
        return access.gate_matrix(bundle.train_xr, bundle.meta.train_comp_id)

    def fit(
        self, bundle: "SurrogateBundle", optimizer: dict, library_specs: list[dict]
    ) -> SINDyBundle:
        preprocessed_xr = transform_gate(
            bundle.preprocessor,
            bundle.train_xr,
            comp_id=bundle.meta.train_comp_id,
        )
        return SINDyBundle.from_sindy(
            library_specs=library_specs,
            optimizer_spec=optimizer,
            x=access.comp_matrix(preprocessed_xr, bundle.meta.train_comp_id),
            u=access.i_ext_values(preprocessed_xr),
            t=access.time(bundle.train_xr),
            targets=[sp.Symbol(v) for v in preprocessed_xr.variable.values],
            inputs=[sp.Symbol("u")],
            # 列構造: [V, g1..gN, u]。V=0, gate 群, 末尾に外部電流。
            roles=Roles(
                V=0,
                g=list(range(1, 1 + bundle.meta.n_components)),
                u=1 + bundle.meta.n_components,
            ),
        )

    def surr_comp_type(self, bundle: "SurrogateBundle") -> CompartmentType:
        xi = jnp.asarray(bundle.sindy_bundle.xi)
        compute_theta = bundle.sindy_bundle.compute_theta()
        n_latent = bundle.meta.n_components

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
            inits=lambda _: (
                [bundle.meta.train_comp.init[0]] + bundle.preprocessor.gate_inits
            ),
            opcost=None,
        )

    def opcost(self, bundle: "SurrogateBundle") -> OpCost:
        return bundle.sindy_bundle.opcost()
