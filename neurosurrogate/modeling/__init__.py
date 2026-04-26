import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .profiler import (
    build_feature_cost_map,
    get_active_features,
    static_calc_cost,
)
from .xarray_utils import set_coords

logger = logging.getLogger(__name__)


class SINDySurrogateWrapper:
    def __init__(self, initialized_sindy, target_module):
        self.sindy = initialized_sindy
        self.target_module = target_module
        self.preprocessor = PCA(n_components=1)

    @staticmethod
    def get_gate_numpy(train_xr, target_comp_id):
        return train_xr["vars"].sel(gate=True, comp_id=target_comp_id).to_numpy()

    def fit(self, train_xr, target_comp_id):
        self.train_gate_data = self.get_gate_numpy(train_xr, target_comp_id)
        self.preprocessor.fit(self.train_gate_data)
        self.preprocessed_xr = self.transform(train_xr, target_comp_id=target_comp_id)
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

    def transform(self, xr_data, target_comp_id):
        xr_gate = self.get_gate_numpy(xr_data, target_comp_id)
        transformed_gate = self.preprocessor.transform(xr_gate)
        v_soma_da = xr_data["vars"].sel(gate=False, comp_id=target_comp_id)
        new_vars = np.concatenate(
            (v_soma_da.to_numpy().reshape(-1, 1), transformed_gate), axis=1
        )

        n_latent = transformed_gate.shape[1]
        coords_config = {
            "comp_id": [target_comp_id] * (n_latent + 1),
            "variable": ["V"] + [f"latent{i + 1}" for i in range(n_latent)],
            "gate": [False] + [True] * n_latent,
        }
        logger.info(coords_config)
        return set_coords(
            raw=new_vars,
            u=xr_data["I_internal"].sel(node_id=target_comp_id).to_numpy(),
            coords=coords_config,
            dt=float(xr_data.time[1] - xr_data.time[0]),
        )

    @property
    def gate_init(self):
        return self.preprocessed_xr["vars"].to_numpy()[0][1:]

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


def get_loggable_summary(
    surrogate: SINDySurrogateWrapper, base_cost_map, original_cost
) -> dict:
    def _format_to_table(cost_map: dict) -> str:
        # 辞書をデータフレームに変換
        df = pd.DataFrame.from_dict(cost_map, orient="index")
        df.index.name = "Feature"
        # 欠損値を0で埋めて整数型にし、美しいMarkdownとして出力
        return df.fillna(0).astype(int).to_markdown()

    coef = surrogate.sindy.optimizer.coef_
    nonzero_term_num = np.count_nonzero(coef)

    feature_cost_map = build_feature_cost_map(
        surrogate.sindy.get_feature_names(), base_cost_map
    )
    active_features_map = build_feature_cost_map(
        get_active_features(surrogate.sindy), base_cost_map
    )

    return {
        "metrics": {
            "nonzero_term_num": str(nonzero_term_num),
            "nonzero_term_ratio": str(nonzero_term_num / coef.size),
            **static_calc_cost(surrogate.sindy, feature_cost_map, original_cost),
            **_get_pca_metrics(surrogate.preprocessor, surrogate.train_gate_data),
        },
        "params": surrogate.sindy.optimizer.get_params(),
        "artifacts": {
            # テキストファイルとして保存するもの (ファイル名: 中身の文字列)
            "texts": {
                "equations.txt": "\n".join(surrogate.sindy.equations(precision=3)),
                "coef.txt": np.array2string(coef, precision=3),
                "features.md": _format_to_table(feature_cost_map),
                "features_active.md": _format_to_table(active_features_map),
                "misc/source.txt": surrogate.source,
            },
            # 画像ファイルとして保存するもの (ファイル名: Figureオブジェクト)
            "xarray": {"train": surrogate.preprocessed_xr},
        },
        "model": {
            "xi": coef,
            "feature_names": surrogate.sindy.get_feature_names(),
            "target_names": surrogate.preprocessed_xr.variable.values.tolist(),
        },
    }


def _get_pca_metrics(pca: PCA, train_gate_data):
    reconstructed = pca.inverse_transform(pca.transform(train_gate_data))
    mse = np.mean((train_gate_data - reconstructed) ** 2)
    return {
        "pca/explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pca/explained_variance": float(pca.explained_variance_[0]),
        "pca/reconstruction_mse": float(mse),
        "pca/reconstruction_mse_ratio": float(mse / np.var(train_gate_data)),
    }
