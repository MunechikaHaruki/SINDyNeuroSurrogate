import logging

import jax.numpy as jnp
import numpy as np

from ..registry.compartments import Compartment

logger = logging.getLogger(__name__)


def _dummy_theta(v, latent, i_int):
    return jnp.zeros(1, dtype=jnp.float64)


_dummy_xi = np.zeros((2, 1), dtype=np.float64)

DUMMY_SINDY_ARGS = (_dummy_xi, _dummy_theta)


def make_surr_comp(name: str, gate_inits: list) -> Compartment:
    return Compartment(
        type_name="surr",
        gate_inits=gate_inits,
        gate_names=[f"latent{i + 1}" for i in range(len(gate_inits))],
        name=name,
    )


def get_gate_numpy(train_xr, target_comp_id):
    return train_xr["vars"].sel(gate=True, comp_id=target_comp_id).to_numpy()


def transform_gate(preprocessor, xr_data, target_comp_id):
    from ..features.coords import StateAccumulator, set_coords

    transformed_gate = preprocessor.transform(get_gate_numpy(xr_data, target_comp_id))
    n_latent = transformed_gate.shape[1]

    return set_coords(
        raw=np.concatenate(
            (
                xr_data["vars"]
                .sel(gate=False, comp_id=target_comp_id)
                .to_numpy()
                .reshape(-1, 1),
                transformed_gate,
            ),
            axis=1,
        ),
        u=xr_data["I_internal"].sel(node_id=target_comp_id).to_numpy(),
        coords=StateAccumulator(
            comp_id=[target_comp_id] * (n_latent + 1),
            variable=["V"] + [f"latent{i + 1}" for i in range(n_latent)],
            gate=[False] + [True] * n_latent,
        ).to_coords(),
        dt=float(xr_data.time[1] - xr_data.time[0]),
    )


class SINDyNeuroSurrogate:
    def __init__(self, preprocessor, initialized_sindy, target_module):
        self.preprocessor = preprocessor
        self.sindy = initialized_sindy
        self.target_module = target_module

    def fit(self, train_xr, target_comp_id) -> None:
        self.train_gate_data = get_gate_numpy(train_xr, target_comp_id)
        self.preprocessor.fit(self.train_gate_data)
        preprocessed_xr = transform_gate(
            self.preprocessor, train_xr, target_comp_id=target_comp_id
        )
        self._gate_inits: list = preprocessed_xr["vars"].to_numpy()[0][1:].tolist()
        self.target_names: list = preprocessed_xr.variable.values.tolist()
        input_features = self.target_names + ["u"]
        self.sindy.fit(
            preprocessed_xr["vars"].sel(comp_id=target_comp_id).to_numpy(),
            u=preprocessed_xr["I_ext"].to_numpy(),
            t=train_xr["time"].to_numpy(),
            feature_names=input_features,
        )
        # 関数のビルド
        self.source = self._build_source(self.sindy.get_feature_names(), input_features)
        logger.info(self.source)
        self.compute_theta = self._compile_source(self.source, self.target_module)

    def make_surr_comp(self, name: str) -> Compartment:
        return make_surr_comp(name, self._gate_inits)

    @property
    def sindy_args(self):
        return (self.sindy.coefficients(), self.compute_theta)

    @staticmethod
    def _compile_source(source, module):
        local_vars = {}
        exec(source, {**vars(module), "jnp": jnp}, local_vars)
        return local_vars["dynamic_compute_theta"]

    @staticmethod
    def _build_source(feature_names: list, input_features: list):
        expressions = []
        for name in feature_names:
            # '1' という文字列は明示的に浮動小数点にする
            safe_name = "1.0" if name == "1" else name
            # SINDyの出力する '^'（べき乗）を Python の '**' に置換
            safe_name = safe_name.replace("^", "**")
            expressions.append(safe_name)
        expr_list = ", ".join(expressions)
        return f"""def dynamic_compute_theta({",".join(input_features)}):
    import jax.numpy as jnp
    return jnp.array([{expr_list}], dtype=jnp.float64)
        """
