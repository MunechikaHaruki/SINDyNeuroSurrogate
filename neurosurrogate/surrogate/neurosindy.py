import logging
from pathlib import Path

import jax.numpy as jnp
import joblib
import numpy as np
import pysindy as ps

from ..core.network import Compartment, CompartmentType
from .libraries import FeatureLibrary

_BUNDLE_FILE = "surrogate.joblib"

logger = logging.getLogger(__name__)


def get_gate_numpy(train_xr, target_comp_id):
    return train_xr["vars"].sel(gate=True, comp_id=target_comp_id).to_numpy()


def transform_gate(preprocessor, xr_data, target_comp_id):
    from ..core.coords import StateAccumulator, set_coords

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


def _build_compute_theta(feature_lib: FeatureLibrary):
    """FeatureLibrary の sub_libraries から compute_theta 関数を直接構築。
    LibraryEntry.func を直接呼び出す → exec / target_module 依存なし。"""
    subs = feature_lib.sub_libraries

    def compute_theta(*inputs):
        values = []
        for sub in subs:
            bound = [inputs[i] for i in sub.inputs]
            for entry in sub.entries:
                values.append(entry.func(*bound))
        return jnp.array(values, dtype=jnp.float64)

    return compute_theta


class SINDyNeuroSurrogate:
    def __init__(self, preprocessor, optimizer, library_specs: list[dict]):
        self.preprocessor = preprocessor
        self.library_specs = library_specs
        self._feature_lib = FeatureLibrary.build(library_specs)
        self.sindy = ps.SINDy(
            feature_library=self._feature_lib.library,
            optimizer=optimizer,
        )

    def fit(self, train_xr, target_comp_id) -> None:
        self.train_comp_id: int = target_comp_id
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
        self.compute_theta = _build_compute_theta(self._feature_lib)
        self.xi: np.ndarray = self.sindy.coefficients()
        self.feature_names: list = self.sindy.get_feature_names()
        self.equations: str = "\n".join(self.sindy.equations(precision=3))

    def make_surr_comp(self, name: str) -> Compartment:
        xi = self.xi
        compute_theta = self.compute_theta

        def surr_kernel(params, i_t, v, state):
            theta = compute_theta(v, state[0], i_t)
            return xi[0] @ theta, jnp.stack([xi[1] @ theta])

        return Compartment(
            name=name,
            type=CompartmentType(
                name="surr",
                kernel=surr_kernel,
                param_cls=None,
                default_params=None,
                gate_names=[f"latent{i + 1}" for i in range(len(self._gate_inits))],
                default_gate_inits=self._gate_inits,
                v_init=-65,
                opcost=None,
            ),
        )

    def save(self, dir: Path | str) -> None:
        bundle = {
            "preprocessor": self.preprocessor,
            "xi": self.xi,
            "gate_inits": self._gate_inits,
            "library_specs": self.library_specs,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "equations": self.equations,
            "train_comp_id": self.train_comp_id,
        }
        joblib.dump(bundle, Path(dir) / _BUNDLE_FILE)

    @classmethod
    def load(cls, dir: Path | str) -> "SINDyNeuroSurrogate":
        bundle = joblib.load(Path(dir) / _BUNDLE_FILE)
        self = cls.__new__(cls)
        self.preprocessor = bundle["preprocessor"]
        self.sindy = None
        self.library_specs = bundle["library_specs"]
        self._feature_lib = FeatureLibrary.build(self.library_specs)
        self.compute_theta = _build_compute_theta(self._feature_lib)
        self.xi = bundle["xi"]
        self._gate_inits = bundle["gate_inits"]
        self.feature_names = bundle["feature_names"]
        self.target_names = bundle["target_names"]
        self.equations = bundle["equations"]
        self.train_comp_id: int = bundle["train_comp_id"]
        return self
