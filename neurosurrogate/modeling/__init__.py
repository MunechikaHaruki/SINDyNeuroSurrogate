import logging

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

from ..utils.plots import plot_compartment_behavior, plot_diff, plot_simple
from .numba_core import unified_simulater

logger = logging.getLogger(__name__)


class PCAPreProcessorWrapper:
    def __init__(self):
        self.pca = PCA(n_components=1)

    def fit(self, train_xr_dataset):
        train_gate_data = train_xr_dataset["vars"].sel(gate=True).to_numpy()
        logger.info("Fitting preprocessor...")
        self.pca.fit(train_gate_data)

    def transform(self, xr_data):
        xr_gate = xr_data["vars"].sel(gate=True).to_numpy()
        transformed_gate = self.pca.transform(xr_gate)
        v_soma_da = xr_data["vars"].sel(variable="V_soma")
        new_vars = np.concatenate(
            (v_soma_da.to_numpy().reshape(-1, 1), transformed_gate), axis=1
        )
        n_latent = transformed_gate.shape[1]
        comp_parts = ["soma"] + ["soma"] * n_latent
        variables = ["V_soma"] + [f"latent{i + 1}" for i in range(n_latent)]
        gate_flags = [False] + [True] * n_latent  # latent も gate 由来なので True

        mindex = pd.MultiIndex.from_arrays(
            [comp_parts, variables, gate_flags],
            names=("compartment", "variable", "gate"),
        )

        # 4. 元の座標と属性を引き継いで DataArray を作成
        return xr.DataArray(
            data=new_vars,
            dims=("time", "features"),
            coords={
                "time": xr_data.time,
                "features": mindex,  # ここで階層構造を復元
            },
            name="vars",
            attrs=xr_data.attrs,
        )


class SINDySurrogateWrapper:
    def __init__(self, preprocessor, target_module, sindy_name):
        self.target_module = target_module
        self.sindy = getattr(target_module, sindy_name)
        self.preprocessor = preprocessor

    def fit(self, train_xr_dataset, direct=False):
        self.train_dataarray = self.preprocessor.transform(train_xr_dataset)
        if direct is True:
            self.u_dataarray = train_xr_dataset["I_internal"].sel(direction="soma")
        elif direct is False:
            self.u_dataarray = train_xr_dataset["I_ext"]

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
        self.compute_theta = self.extract_compute_theta_from_sindy(
            self.sindy, self.target_module
        )

    def predict(self, init, dt, u, data_type, params_dict):
        logger.info(f"{data_type}のサロゲートモデルをテスト")
        if data_type == "hh3":
            init = np.array([init[0], init[1], -65, -65])

        return unified_simulater(
            dt=dt,
            u=u,
            data_type=data_type,
            params_dict=params_dict,
            mode="surrogate",
            init=init,
            xi=self.sindy.coefficients(),
            compute_theta=self.compute_theta,
        )

    def eval(self, original_ds):
        transformed_dataarray = self.preprocessor.transform(original_ds)
        predict_result = self.predict(
            init=transformed_dataarray[0].to_numpy(),
            dt=float(original_ds.attrs["dt"]),
            u=original_ds["I_ext"].to_numpy(),
            data_type=original_ds.attrs["model_type"],
            params_dict=original_ds.attrs["params"],
        )

        if original_ds.attrs["model_type"] == "hh":
            u_inj = original_ds["I_ext"].to_numpy()
        elif original_ds.attrs["model_type"] == "hh3":
            u_inj = original_ds["I_internal"].sel(direction="soma")

        return {
            "surrogate_figure": plot_simple(predict_result),
            "diff": plot_diff(
                original=original_ds,
                preprocessed=transformed_dataarray,
                surrogate=predict_result,
            ),
            "preprocessed": plot_compartment_behavior(
                u=u_inj, xarray=transformed_dataarray
            ),
        }

    @staticmethod
    def extract_compute_theta_from_sindy(sindy_model, target_module):
        """
        SINDyオブジェクトから特徴量計算式を抽出し、
        Numbaでコンパイルされた関数を生成する。
        """
        feature_names = sindy_model.get_feature_names()
        input_features = ",".join(sindy_model.feature_names)
        # ソースコードの組み立て
        array_content = ",\n".join(feature_names)
        source = f"""
@njit
def dynamic_compute_theta({input_features}):
    return np.array([
        {array_content}
    ]
    )
"""
        logger.info(source)
        # 実行環境のglobals()を引き継ぎ、alpha_mなどの関数を参照可能にする
        local_vars = {}
        exec(source, vars(target_module), local_vars)
        return local_vars["dynamic_compute_theta"]

    @staticmethod
    def static_calc_cost(sindy_model):
        """ "
        expの演算回数が~~~,+の演算回数が~~~みたいに計算
        """

        result = {"exp": 0, "pow": 0, "dot": 0, "plus": 0, "divide": 0}

        cost_map = {
            "alpha": {
                "exp": 1,
                "+": 2,
            },
            "beta": {},
        }

        # 係数が非ゼロのインデックスを取得
        coefs = sindy_model.coefficients()
        # 各変数（v, m, h等）の微分方程式において、一つでも非ゼロ係数を持つ項のマスク
        active_mask = np.any(coefs != 0, axis=0)

        # すべての特徴量名を取得し、アクティブなものだけに絞り込む
        all_features = sindy_model.get_feature_names()
        active_features = [f for i, f in enumerate(all_features) if active_mask[i]]

        for name in active_features:
            pass

        return result

    def get_loggable_summary(self) -> dict:
        return {
            "equations": self.sindy.equations(precision=3),
            "coefficients": self.sindy.optimizer.coef_,
            "feature_names": self.sindy.get_feature_names(),
            "model_params": str(self.sindy.optimizer.get_params),
            "train_figure": plot_compartment_behavior(
                xarray=self.train_dataarray, u=self.u_dataarray
            ),
            "static_calc_cost": self.static_calc_cost(self.sindy),
        }
