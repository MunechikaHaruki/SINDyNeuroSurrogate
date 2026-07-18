import jax.numpy as jnp
import sympy as sp

from ...core import access
from ...core.coords import transform_gate
from ...core.network import Compartment, CompartmentType
from ...core.opcost import OpCost
from ..bundle import PreprocessorBundle, SINDyBundle
from ..replace import resolved_params
from .base import NeuroSurrogateBase
from .roles import Roles


class SINDyNeuroSurrogate(NeuroSurrogateBase):
    SURROGATE_TYPE = "sindy"

    def fit(self, optimizer, library_specs: list[dict]) -> None:
        train_gate = access.gate_matrix(self._train_xr, self._meta.train_comp_id)
        preprocessor_bundle = PreprocessorBundle.from_spec(
            {**self._preprocessor, "n_components": self._meta.n_components}, train_gate
        )
        preprocessed_xr = transform_gate(
            preprocessor_bundle.preprocessor,
            self._train_xr,
            comp_id=self._meta.train_comp_id,
        )
        self._set_bundles(
            sindy_bundle=SINDyBundle.from_sindy(
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
            ),
            preprocessor_bundle=preprocessor_bundle,
        )

    def params_compatible(self, comp: Compartment) -> bool:
        # surr は param_cls=None → simulator がノード params を捨て、学習モデルが
        # V+gate 全体を train params 込みで再現。→ 全 params 完全一致が必須。
        return resolved_params(self.meta.train_comp) == resolved_params(comp)

    @property
    def surr_comp_type(self) -> CompartmentType:
        xi = jnp.asarray(self.sindy_bundle.xi)
        compute_theta = self.sindy_bundle.compute_theta()
        n_latent = len(self.preprocessor_bundle.gate_inits)

        def surr_kernel(params, i_t, v, state):
            # 列構造 [V, g1..gN, u] の順で束縛。xi の行も同順 (0=V, 1..=latent)。
            theta = compute_theta(v, *(state[i] for i in range(n_latent)), i_t)
            return xi[0] @ theta, xi[1:] @ theta

        return CompartmentType(
            name="surr",
            kernel=surr_kernel,
            param_cls=None,
            gate_names=access.latent_vars(n_latent),
            default_gate_inits=self.preprocessor_bundle.gate_inits,
            v_init=-65,
            opcost=None,
        )

    @property
    def opcost(self) -> OpCost:
        return self.sindy_bundle.opcost()
