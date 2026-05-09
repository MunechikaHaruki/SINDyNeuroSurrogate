import json
from dataclasses import dataclass, fields

import numpy as np
from sklearn.decomposition import PCA

from .model_neurosindy import SINDyResult


@dataclass(frozen=True)
class OpCost:
    exp: int = 0
    div: int = 0
    pm: int = 0
    mul: int = 0

    def __add__(self, other: "OpCost") -> "OpCost":
        return OpCost(
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in fields(self)
            }
        )

    def __mul__(self, n: int) -> "OpCost":
        return OpCost(**{f.name: getattr(self, f.name) * n for f in fields(self)})

    def to_dict(self) -> dict[str, int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


HH_RATE_COST_MAP: dict[str, OpCost] = {
    "alpha_m": OpCost(exp=1, div=1, pm=2, mul=2),
    "beta_m": OpCost(exp=1, div=1, pm=1, mul=1),
    "alpha_h": OpCost(exp=1, div=1, pm=1, mul=1),
    "beta_h": OpCost(exp=1, div=1, pm=2, mul=1),
    "alpha_n": OpCost(exp=1, div=1, pm=2, mul=2),
    "beta_n": OpCost(exp=1, div=1, pm=1, mul=1),
}


HH_COST = (
    sum(HH_RATE_COST_MAP.values(), OpCost())  # レート関数
    + OpCost(pm=1)  # 反転電位
    + OpCost(pm=3, mul=5) * 2  # Na,K電流
    + OpCost(pm=1, mul=1)  # leak電流
    + OpCost(pm=6, mul=6)  # dg/dt
    + OpCost(pm=3, div=1)  # dv/dtの計算
)


def calc_preprocessor_metrics(preprocessor, train_gate_data: np.ndarray):
    def _get_pca_metrics(pca: PCA, train_gate_data):
        reconstructed = pca.inverse_transform(pca.transform(train_gate_data))
        mse = np.mean((train_gate_data - reconstructed) ** 2)
        return {
            "pca/explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "pca/explained_variance": float(pca.explained_variance_[0]),
            "pca/reconstruction_mse": float(mse),
            "pca/reconstruction_mse_ratio": float(mse / np.var(train_gate_data)),
        }

    if isinstance(preprocessor, PCA):
        preprocessor_metrics = _get_pca_metrics(preprocessor, train_gate_data)
    else:
        preprocessor_metrics = {}
    return preprocessor_metrics


@dataclass(frozen=True)
class SINDyAnalyzer:
    result: SINDyResult
    feature_cost_map: dict[str, OpCost]
    original_cost: OpCost

    @property
    def _active_features(self):
        # 係数が非ゼロのインデックスを取得
        # 各変数（v, m, h等）の微分方程式において、一つでも非ゼロ係数を持つ項のマスク
        active_mask = np.any(self.result.coef != 0, axis=0)
        # すべての特徴量名を取得し、アクティブなものだけに絞り込む
        _active_features = [
            f for i, f in enumerate(self.feature_cost_map.keys()) if active_mask[i]
        ]
        return _active_features

    @property
    def _active_features_map(self):
        return {k: self.feature_cost_map[k] for k in self._active_features}

    @property
    def _stat_calc_cost(self):
        nnz = np.count_nonzero(self.result.coef).item()
        surrogate = OpCost(mul=nnz, pm=max(0, nnz - int(self.result.coef.shape[0])))

        for feature in self._active_features:
            if feature not in self.feature_cost_map:
                raise ValueError(
                    f"未知の基底関数 '{feature}' が見つかりました。"
                    "注入された cost_map にこの基底関数の定義を追加してください。"
                )
            surrogate = surrogate + self.feature_cost_map[feature]

        surr_d = surrogate.to_dict()
        orig_d = self.original_cost.to_dict()
        return {
            **{f"cost/surrogate/{k}": v for k, v in surr_d.items()},
            **{f"cost/original/{k}": v for k, v in orig_d.items()},
            **{f"cost/diff/{k}": surr_d[k] - orig_d[k] for k in orig_d},
        }

    @property
    def metrics(self) -> dict[str, float]:
        nonzero_term_num = np.count_nonzero(self.result.coef)
        return {
            "nonzero_term_num": int(nonzero_term_num),
            "nonzero_term_ratio": float(nonzero_term_num / self.result.coef.size),
            **self._stat_calc_cost,
            **calc_preprocessor_metrics(
                self.result.preprocessor, self.result.train_gate_data
            ),
        }

    @property
    def texts(self) -> dict[str, str]:
        return {
            "equations.txt": self.result.equations,
            "coef.txt": np.array2string(self.result.coef, precision=3),
            "features.json": json.dumps(
                {k: v.to_dict() for k, v in self.feature_cost_map.items()}
            ),
            "features_active.json": json.dumps(
                {k: v.to_dict() for k, v in self._active_features_map.items()}
            ),
            "misc/source.txt": self.result.source,
        }

    @property
    def view(self):
        return {
            "xi_matrix": self.result.coef.tolist(),
            "feature_names": self.feature_cost_map.keys(),
            "target_names": self.result.target_names,
        }

    @property
    def params(self):
        return self.result.params
