import logging

import numpy as np
from numba import njit

from .xarray_utils import get_gate_numpy, transform_gate

logger = logging.getLogger(__name__)


@njit
def _dummy_theta(v, latent, i_int):
    return np.zeros(1, dtype=np.float64)


_dummy_xi = np.zeros((2, 1), dtype=np.float64)

DUMMY_SINDY_ARGS = (_dummy_xi, _dummy_theta)
DUMMY_SURR_COMP = {
    "init": np.array([0, 0]),
    "vars": ["V"] + [f"latent{i + 1}" for i in range(1)],
    "gate": [False] + [True],
}


class SINDyNeuroSurrogate:
    def __init__(self, preprocessor, initialized_sindy, target_module):
        self.preprocessor = preprocessor
        self.sindy = initialized_sindy
        self.target_module = target_module

    def fit(self, train_xr, target_comp_id):
        self.train_gate_data = get_gate_numpy(train_xr, target_comp_id)
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
        self.source = self._build_source(self.sindy.get_feature_names(), input_features)
        logger.info(self.source)
        self.compute_theta = self._compile_source(self.source, self.target_module)

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
    def _compile_source(source, module):
        local_vars = {}
        exec(source, vars(module), local_vars)
        return local_vars["dynamic_compute_theta"]

    @staticmethod
    def _build_source(feature_names: list, input_features: list):
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
        # 3. テンプレートを組み立て
        return f"""@njit
def dynamic_compute_theta({",".join(input_features)}):
    res = np.empty({num_features}, dtype=np.float64)
{array_content}
    return res
        """
