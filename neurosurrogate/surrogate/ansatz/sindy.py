import jax.numpy as jnp

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

    def fit(self, preprocessor, optimizer, library_specs: list[dict]) -> None:
        train_gate = access.gate_matrix(self._train_xr, self._meta.train_comp_id)
        preprocessor_bundle = PreprocessorBundle.from_spec(
            {**preprocessor, "n_components": self._n_components}, train_gate
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
                target_names=preprocessed_xr.variable.values.tolist(),
                input_names=["u"],
                # 列構造: [V, latent1..N, u]。V=0, gate 群, 末尾に外部電流。
                roles=Roles(
                    V=0,
                    g=list(range(1, 1 + self._n_components)),
                    u=1 + self._n_components,
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
        xi = self.sindy_bundle.xi
        compute_theta = self.sindy_bundle.compute_theta()

        def surr_kernel(params, i_t, v, state):
            theta = compute_theta(v, state[0], i_t)
            return xi[0] @ theta, jnp.stack([xi[1] @ theta])

        return CompartmentType(
            name="surr",
            kernel=surr_kernel,
            param_cls=None,
            gate_names=[
                f"latent{i + 1}"
                for i in range(len(self.preprocessor_bundle.gate_inits))
            ],
            default_gate_inits=self.preprocessor_bundle.gate_inits,
            v_init=-65,
            opcost=None,
        )

    @property
    def opcost(self) -> OpCost:
        return self.sindy_bundle.opcost()
