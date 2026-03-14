import logging

import numpy as np
from sklearn.decomposition import PCA

from ..utils.plots import plot_compartment_behavior
from .profiler import build_feature_cost_map, get_active_features, static_calc_cost
from .xarray_utils import generate_preprocessed_xarray

logger = logging.getLogger(__name__)


class PCAPreProcessorWrapper:
    def __init__(self):
        self.pca = PCA(n_components=1)

    def fit(self, train_xr_dataset, target_comp_id):
        train_gate_data = (
            train_xr_dataset["vars"].sel(gate=True, comp_id=target_comp_id).to_numpy()
        )
        logger.info("Fitting preprocessor...")
        self.pca.fit(train_gate_data)

    def transform(self, xr_data, target_comp_id):
        xr_gate = xr_data["vars"].sel(gate=True, comp_id=target_comp_id).to_numpy()
        transformed_gate = self.pca.transform(xr_gate)
        v_soma_da = xr_data["vars"].sel(gate=False, comp_id=target_comp_id)
        new_vars = np.concatenate(
            (v_soma_da.to_numpy().reshape(-1, 1), transformed_gate), axis=1
        )
        n_latent = transformed_gate.shape[1]
        return generate_preprocessed_xarray(new_vars, xr_data.time, n_latent)


class SINDySurrogateWrapper:
    def __init__(self, initialized_sindy, target_module, base_cost_map, original_cost):
        self.sindy = initialized_sindy
        self.target_module = target_module
        self.base_cost_map = base_cost_map
        self.original_cost = original_cost

        self.preprocessor = PCAPreProcessorWrapper()

    def fit(self, train_xr_dataset, target_comp_id):
        self.preprocessor.fit(train_xr_dataset, target_comp_id=target_comp_id)

        self.train_dataarray = self.preprocessor.transform(
            train_xr_dataset, target_comp_id=target_comp_id
        )
        self.u_dataarray = train_xr_dataset["I_internal"].sel(node_id=target_comp_id)

        input_features = self.train_dataarray.get_index("features").get_level_values(
            "variable"
        ).tolist() + ["u"]
        logger.critical(input_features)

        self.sindy.fit(
            self.train_dataarray.to_numpy(),
            u=self.u_dataarray.to_numpy(),
            t=train_xr_dataset["time"].to_numpy(),
            feature_names=input_features,
        )
        self.compute_theta = extract_compute_theta_from_sindy(
            self.sindy, self.target_module
        )

        self.gate_init = self.train_dataarray.to_numpy()[0][1:]

        self.feature_cost_map = build_feature_cost_map(
            self.sindy.get_feature_names(), self.base_cost_map
        )

    def get_loggable_summary(self) -> dict:
        return {
            "equations": self.sindy.equations(precision=3),
            "coefficients": self.sindy.optimizer.coef_,
            "feature_names": self.sindy.get_feature_names(),
            "active_features": get_active_features(self.sindy),
            "model_params": str(self.sindy.optimizer.get_params),
            "train_figure": plot_compartment_behavior(
                xarray=self.train_dataarray, u=self.u_dataarray
            ),
            "feature_cost_map": self.feature_cost_map,
            "static_calc_cost": static_calc_cost(
                self.sindy, self.feature_cost_map, self.original_cost
            ),
        }


def extract_compute_theta_from_sindy(sindy_model, target_module):
    """
    SINDyオブジェクトから特徴量計算式を抽出し、
    Numbaでコンパイルされた関数を生成する。
    """
    feature_names = sindy_model.get_feature_names()
    input_features = ",".join(sindy_model.feature_names)
    # ソースコードの組み立て
    array_content = ",\n".join(feature_names)
    source = f"""@njit
def dynamic_compute_theta({input_features}):
    return np.array([
{array_content}])"""
    logger.info(source)
    # 実行環境のglobals()を引き継ぎ、alpha_mなどの関数を参照可能にする
    local_vars = {}
    exec(source, vars(target_module), local_vars)
    return local_vars["dynamic_compute_theta"]
