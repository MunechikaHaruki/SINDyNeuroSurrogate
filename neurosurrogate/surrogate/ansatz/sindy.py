import jax.numpy as jnp
import numpy as np
import sympy as sp

from ...core import access
from ...core.coords import transform_gate
from ...core.network import Compartment, CompartmentType
from ...core.opcost import OpCost
from ..bundle import SINDyBundle
from .base import NeuroSurrogateBase
from .roles import Roles


class SINDyNeuroSurrogate(NeuroSurrogateBase):
    SURROGATE_TYPE = "sindy"

    def _train_gate(self) -> np.ndarray:
        return access.gate_matrix(self._train_xr, self._meta.train_comp_id)

    def fit(self, optimizer, library_specs: list[dict]) -> None:
        preprocessed_xr = transform_gate(
            self.preprocessor,
            self._train_xr,
            comp_id=self._meta.train_comp_id,
        )
        self._sindy_bundle = SINDyBundle.from_sindy(
            library_specs=library_specs,
            optimizer_spec=optimizer,
            x=access.comp_matrix(preprocessed_xr, self._meta.train_comp_id),
            u=access.i_ext_values(preprocessed_xr),
            t=access.time(self._train_xr),
            targets=[sp.Symbol(v) for v in preprocessed_xr.variable.values],
            inputs=[sp.Symbol("u")],
            # 列構造: [V, g1..gN, u]。V=0, gate 群, 末尾に外部電流。
            roles=Roles(
                V=0,
                g=list(range(1, 1 + self._meta.n_components)),
                u=1 + self._meta.n_components,
            ),
        )

    def params_compatible(self, comp: Compartment) -> bool:
        # surr は param_cls=None → simulator がノード params を捨て、学習モデルが
        # V+gate 全体を train params 込みで再現。→ 全 params 完全一致が必須。
        return self.meta.train_comp.resolved_params == comp.resolved_params

    @property
    def surr_comp_type(self) -> CompartmentType:
        xi = jnp.asarray(self.sindy_bundle.xi)
        compute_theta = self.sindy_bundle.compute_theta()
        n_latent = self._meta.n_components

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
                [self.meta.train_comp.init[0]] + self.preprocessor.gate_inits
            ),
            opcost=None,
        )

    @property
    def opcost(self) -> OpCost:
        return self.sindy_bundle.opcost()
