import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..utils.plots import plot_compartment_behavior, plot_diff, plot_simple
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

        self.gate_init = self.train_dataarray.to_numpy()[0][1:]

        # 関数組み立て
        array_content = ",\n".join(self.sindy.get_feature_names())
        input_features = ",".join(self.sindy.feature_names)
        self.source = f"""@njit
def dynamic_compute_theta({input_features}):
    return np.array([
{array_content}])"""
        local_vars = {}
        exec(self.source, vars(self.target_module), local_vars)
        self.compute_theta = local_vars["dynamic_compute_theta"]

        logger.info(self.source)

    def get_loggable_summary(self) -> dict:
        coef = self.sindy.optimizer.coef_
        nonzero_term_num = np.count_nonzero(coef)

        feature_cost_map = build_feature_cost_map(
            self.sindy.get_feature_names(), self.base_cost_map
        )
        active_features_map = build_feature_cost_map(
            get_active_features(self.sindy), self.base_cost_map
        )

        return {
            "metrics": {
                **static_calc_cost(self.sindy, feature_cost_map, self.original_cost),
                "nonzero_term_num": int(nonzero_term_num),
                "nonzero_term_ratio": float(nonzero_term_num / coef.size),
            },
            "params": self.sindy.optimizer.get_params(),
            "artifacts": {
                # テキストファイルとして保存するもの (ファイル名: 中身の文字列)
                "texts": {
                    "equations.txt": "\n".join(self.sindy.equations(precision=3)),
                    "source.txt": self.source,
                    "coef.txt": np.array2string(coef, precision=3),
                    "features.md": self._format_to_table(feature_cost_map),
                    "features_active.md": self._format_to_table(active_features_map),
                },
                # 画像ファイルとして保存するもの (ファイル名: Figureオブジェクト)
                "figures": {
                    "train.png": plot_compartment_behavior(
                        xarray=self.train_dataarray, u=self.u_dataarray
                    )
                },
            },
        }

    @staticmethod
    def _format_to_table(cost_map: dict) -> str:
        # 辞書をデータフレームに変換
        df = pd.DataFrame.from_dict(cost_map, orient="index")
        df.index.name = "Feature"
        # 欠損値を0で埋めて整数型にし、美しいMarkdownとして出力
        return df.fillna(0).astype(int).to_markdown()


def analyze_eval_results(
    original_ds, predict_result, name, target_comp_id, surrogate_model
):
    """
    シミュレーション済みのデータを受け取り、メトリクス計算と可視化を行う。
    シミュレーター自体は呼び出さない。
    """
    data_type = original_ds.attrs["model_type"]
    dt = float(original_ds.attrs["dt"])

    # 1. 前処理済みデータの取得
    transformed_dataarray = surrogate_model.preprocessor.transform(
        original_ds, target_comp_id=target_comp_id
    )

    # 2. メトリクス計算 (ISI等)
    # orig_v = original_ds["v"].sel(node_id=target_comp_id).to_numpy()
    # surr_v = predict_result["v"].sel(node_id=target_comp_id).to_numpy()
    # dynamic_metrics = calc_dynamic_metrics(orig_v, surr_v, dt)
    dynamic_metrics = {}  # dummy

    def set_prefix_to_metrics(metrics: dict):
        return {f"eval/{data_type}/{name}/{k}": v for k, v in metrics.items()}

    # 3. 構造化して返す
    return {
        "metrics": set_prefix_to_metrics(dynamic_metrics),
        "figures": {
            f"preprocessed/{data_type}/{name}.png": plot_compartment_behavior(
                u=original_ds["I_internal"].sel(node_id=target_comp_id),
                xarray=transformed_dataarray,
            ),
            f"surrogate/{data_type}/{name}.png": plot_simple(predict_result),
            f"compare/{data_type}/{name}.png": plot_diff(
                original=original_ds,
                preprocessed=transformed_dataarray,
                surrogate=predict_result,
            ),
        },
    }
