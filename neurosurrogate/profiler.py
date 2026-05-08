import json
from dataclasses import dataclass, fields

import numpy as np
from sklearn.decomposition import PCA

from .model import SINDyResult


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


def get_active_features(coef, base_names):
    # 係数が非ゼロのインデックスを取得
    # 各変数（v, m, h等）の微分方程式において、一つでも非ゼロ係数を持つ項のマスク
    active_mask = np.any(coef != 0, axis=0)

    # すべての特徴量名を取得し、アクティブなものだけに絞り込む
    active_features = [f for i, f in enumerate(base_names) if active_mask[i]]
    return active_features


def stat_calc_cost(
    coef, base_names, cost_map: dict[str, OpCost], original_cost: OpCost
):
    nnz = np.count_nonzero(coef).item()
    surrogate = OpCost(mul=nnz, pm=max(0, nnz - int(coef.shape[0])))

    for feature in get_active_features(coef, base_names):
        if feature not in cost_map:
            raise ValueError(
                f"未知の基底関数 '{feature}' が見つかりました。"
                "注入された cost_map にこの基底関数の定義を追加してください。"
            )
        surrogate = surrogate + cost_map[feature]

    surr_d = surrogate.to_dict()
    orig_d = original_cost.to_dict()
    return {
        **{f"cost/surrogate/{k}": v for k, v in surr_d.items()},
        **{f"cost/original/{k}": v for k, v in orig_d.items()},
        **{f"cost/diff/{k}": surr_d[k] - orig_d[k] for k in orig_d},
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


@dataclass
class SurrogateSummary:
    metrics: dict[str, float]
    params: dict
    view: dict
    texts: dict[str, str]  # filename -> content


def get_loggable_summary(
    result: SINDyResult, original_cost: OpCost, feature_cost_map: dict[str, OpCost]
) -> SurrogateSummary:
    nonzero_term_num = np.count_nonzero(result.coef)
    active_features = get_active_features(result.coef, result.base_names)
    active_features_map = {k: feature_cost_map[k] for k in active_features}

    if isinstance(result.preprocessor, PCA):
        preprocessor_metrics = _get_pca_metrics(
            result.preprocessor, result.train_gate_data
        )
    else:
        preprocessor_metrics = {}

    return SurrogateSummary(
        metrics={
            "nonzero_term_num": int(nonzero_term_num),
            "nonzero_term_ratio": float(nonzero_term_num / result.coef.size),
            **stat_calc_cost(
                result.coef,
                result.base_names,
                feature_cost_map,
                original_cost,
            ),
            **preprocessor_metrics,
        },
        params=result.params,
        view={
            "xi_matrix": result.coef.tolist(),
            "feature_names": result.base_names,
            "target_names": result.target_names,
        },
        texts={
            "equations.txt": result.equations,
            "coef.txt": np.array2string(result.coef, precision=3),
            "features.json": json.dumps(
                {k: v.to_dict() for k, v in feature_cost_map.items()}
            ),
            "features_active.json": json.dumps(
                {k: v.to_dict() for k, v in active_features_map.items()}
            ),
            "misc/source.txt": result.source,
        },
    )
