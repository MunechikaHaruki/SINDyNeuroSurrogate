import logging

import numpy as np
from numba import njit

from .xarray_utils import StateAccumulator, set_coords

logger = logging.getLogger(__name__)


class DummySurrogate:
    @njit
    def dummy_theta(v, latent, i_int):
        return np.zeros(1, dtype=np.float64)

    @property
    def sindy_args(self):
        dummy_xi = np.zeros((2, 1), dtype=np.float64)
        return (dummy_xi, self.dummy_theta)

    @property
    def surr_comp(self):
        return {
            "init": np.array([0, 0]),
            "vars": ["V"] + [f"latent{i + 1}" for i in range(1)],
            "gate": [False] + [True],
        }


class SINDyNeuroSurrogate:
    def __init__(self, preprocessor, initialized_sindy, target_module):
        self.preprocessor = preprocessor
        self.sindy = initialized_sindy
        self.target_module = target_module

    @staticmethod
    def get_gate_numpy(train_xr, target_comp_id):
        return train_xr["vars"].sel(gate=True, comp_id=target_comp_id).to_numpy()

    def fit(self, train_xr, target_comp_id):
        self.train_gate_data = self.get_gate_numpy(train_xr, target_comp_id)
        self.preprocessor.fit(self.train_gate_data)
        self.preprocessed_xr = transform_gate(
            self.preprocessor, train_xr, target_comp_id=target_comp_id
        )
        input_features = self.preprocessed_xr.variable.values.tolist() + ["u"]
        logger.info(input_features)
        self.sindy.fit(
            self.preprocessed_xr["vars"].sel(comp_id=target_comp_id).to_numpy(),
            u=self.preprocessed_xr["I_ext"].to_numpy(),
            t=train_xr["time"].to_numpy(),
            feature_names=input_features,
        )
        # 関数のビルド
        self.source = self._build_source(self.sindy)
        local_vars = {}
        exec(self.source, vars(self.target_module), local_vars)
        self.compute_theta = local_vars["dynamic_compute_theta"]
        logger.info(self.source)

    @property
    def surr_comp(self):
        full_init = self.preprocessed_xr["vars"].to_numpy()[0]
        num_latents = len(self.preprocessed_xr["vars"].to_numpy()[0][1:])
        return {
            "init": full_init,
            "vars": ["V"] + [f"latent{i + 1}" for i in range(num_latents)],
            "gate": [False] + [True] * num_latents,
        }

    @property
    def sindy_args(self):
        return (self.sindy.coefficients(), self.compute_theta)

    @staticmethod
    def _build_source(sindy):
        # 関数組み立て
        feature_names = sindy.get_feature_names()
        num_features = len(feature_names)

        # 各要素を res[i] = ... の形に変換
        assignments = []
        for i, name in enumerate(feature_names):
            # '1' という文字列が来た場合は、Numbaの型推論を助けるために '1.0' に置換
            safe_name = "1.0" if name == "1" else name
            # SINDyの出力する '^'（べき乗）を Python の '**' に置換
            safe_name = safe_name.replace("^", "**")
            assignments.append(f"    res[{i}] = {safe_name}")

        array_content = "\n".join(assignments)
        input_features = ",".join(sindy.feature_names)

        # 3. テンプレートを組み立て
        return f"""@njit
def dynamic_compute_theta({input_features}):
    res = np.empty({num_features}, dtype=np.float64)
{array_content}
    return res
        """


def transform_gate(preprocessor, xr_data, target_comp_id):
    xr_gate = SINDyNeuroSurrogate.get_gate_numpy(xr_data, target_comp_id)
    transformed_gate = preprocessor.transform(xr_gate)
    v_soma_da = xr_data["vars"].sel(gate=False, comp_id=target_comp_id)
    new_vars = np.concatenate(
        (v_soma_da.to_numpy().reshape(-1, 1), transformed_gate), axis=1
    )

    n_latent = transformed_gate.shape[1]

    coords = StateAccumulator(
        comp_id=[target_comp_id] * (n_latent + 1),
        variable=["V"] + [f"latent{i + 1}" for i in range(n_latent)],
        gate=[False] + [True] * n_latent,
    ).to_coords()

    return set_coords(
        raw=new_vars,
        u=xr_data["I_internal"].sel(node_id=target_comp_id).to_numpy(),
        coords=coords,
        dt=float(xr_data.time[1] - xr_data.time[0]),
    )
