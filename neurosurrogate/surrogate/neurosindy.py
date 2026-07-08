import logging
from pathlib import Path

import jax.numpy as jnp
import joblib
import numpy as np

from ..core.network import Compartment, CompartmentType
from .libraries import FeatureLibrary

_BUNDLE_FILE = "surrogate.joblib"

logger = logging.getLogger(__name__)


def _dummy_theta(v, latent, i_int):
    return jnp.zeros(1, dtype=jnp.float64)


_dummy_xi = np.zeros((2, 1), dtype=np.float64)

DUMMY_SINDY_ARGS = (_dummy_xi, _dummy_theta)


def _make_surr_kernel(xi_matrix, compute_theta):
    """統一 signature (params, i_t, v, state) -> (dv, dstate) の SINDy kernel closure"""

    def surr_kernel(params, i_t, v, state):
        # state: shape (n_latent,)、現状 n_latent=1
        theta = compute_theta(v, state[0], i_t)
        dv = xi_matrix[0] @ theta
        dlat = xi_matrix[1] @ theta
        return dv, jnp.stack([dlat])

    return surr_kernel


def make_surr_type(gate_inits, xi_matrix, compute_theta) -> CompartmentType:
    return CompartmentType(
        name="surr",
        kernel=_make_surr_kernel(xi_matrix, compute_theta),
        param_cls=None,
        default_params=None,
        gate_names=[f"latent{i + 1}" for i in range(len(gate_inits))],
        default_gate_inits=gate_inits,
        v_init=-65,
        opcost=None,
    )


def make_surr_comp(
    name: str, gate_inits: list, xi_matrix, compute_theta
) -> Compartment:
    return Compartment(
        name=name,
        type=make_surr_type(gate_inits, xi_matrix, compute_theta),
    )


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
    def __init__(self, preprocessor, initialized_sindy, library_specs: list[dict]):
        self.preprocessor = preprocessor
        self.sindy = initialized_sindy
        self.library_specs = library_specs

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
        # FeatureLibrary から直接 compute_theta 構築 (exec なし)
        feature_lib = FeatureLibrary.build(self.library_specs)
        self.compute_theta = _build_compute_theta(feature_lib)
        # save 用スナップショット (fit 後は self.sindy に触れない)
        self.xi: np.ndarray = self.sindy.coefficients()
        self.feature_names: list = self.sindy.get_feature_names()
        self.equations: str = "\n".join(self.sindy.equations(precision=3))

    def make_surr_comp(self, name: str) -> Compartment:
        xi, theta = self.sindy_args
        return make_surr_comp(name, self._gate_inits, xi, theta)

    @property
    def sindy_args(self):
        return (self.xi, self.compute_theta)

    def save(self, dir: Path | str, extra: dict | None = None) -> None:
        """dir 直下に surrogate.joblib 1 ファイルを書き出し。extra は任意メタ。"""
        bundle = {
            "preprocessor": self.preprocessor,
            "xi": self.xi,
            "gate_inits": self._gate_inits,
            "library_specs": self.library_specs,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "equations": self.equations,
            "extra": extra or {},
        }
        joblib.dump(bundle, Path(dir) / _BUNDLE_FILE)

    @classmethod
    def load(cls, dir: Path | str) -> "SINDyNeuroSurrogate":
        """dir/surrogate.joblib から SINDyNeuroSurrogate 復元。
        pysindy 学習オブジェクト (self.sindy) は復元不要 → None。
        compute_theta は library_specs から FeatureLibrary 再構築 → 直接構築。
        extra メタは self.manifest_extra に保持。"""
        bundle = joblib.load(Path(dir) / _BUNDLE_FILE)
        self = cls.__new__(cls)
        self.preprocessor = bundle["preprocessor"]
        self.sindy = None
        self.library_specs = bundle["library_specs"]
        feature_lib = FeatureLibrary.build(self.library_specs)
        self.compute_theta = _build_compute_theta(feature_lib)
        self.xi = bundle["xi"]
        self._gate_inits = bundle["gate_inits"]
        self.feature_names = bundle["feature_names"]
        self.target_names = bundle["target_names"]
        self.equations = bundle["equations"]
        self.manifest_extra = bundle["extra"]
        return self
