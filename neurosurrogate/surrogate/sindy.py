import jax.numpy as jnp

from ..core.network import CompartmentType
from ..core.opcost import OpCost
from . import get_gate_numpy, transform_gate
from .base import NeuroSurrogateBase
from .bundle import PreprocessorBundle, SINDyBundle


class SINDyNeuroSurrogate(NeuroSurrogateBase):
    SURROGATE_TYPE = "sindy"

    def fit(self, preprocessor, optimizer, library_specs: list[dict]) -> None:
        train_gate = get_gate_numpy(self._train_xr, self._meta.train_comp_id)
        preprocessor_bundle = PreprocessorBundle.from_spec(preprocessor, train_gate)
        preprocessed_xr = transform_gate(
            preprocessor_bundle.preprocessor,
            self._train_xr,
            target_comp_id=self._meta.train_comp_id,
        )
        target_names = preprocessed_xr.variable.values.tolist()
        self._set_bundles(
            sindy_bundle=SINDyBundle.from_sindy(
                library_specs=library_specs,
                optimizer_spec=optimizer,
                x=preprocessed_xr["vars"]
                .sel(comp_id=self._meta.train_comp_id)
                .to_numpy(),
                u=preprocessed_xr["I_ext"].to_numpy(),
                t=self._train_xr["time"].to_numpy(),
                target_names=target_names,
                input_names=["u"],
            ),
            preprocessor_bundle=preprocessor_bundle,
        )

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
            default_params=None,
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
