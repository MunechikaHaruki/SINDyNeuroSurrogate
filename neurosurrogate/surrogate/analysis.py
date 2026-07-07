import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import PCA

from ..opcost import OpCost

if TYPE_CHECKING:
    from .neurosindy import SINDyNeuroSurrogate


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
        return _get_pca_metrics(preprocessor, train_gate_data)
    return {}


@dataclass(frozen=True)
class SINDyResult:
    coef: np.ndarray
    feature_names: list[str]
    equations: str
    source: str
    target_names: list[str]
    params: dict
    preprocessor_metrics: dict[str, float]


@dataclass(frozen=True)
class SINDySummary:
    metrics: dict[str, float]
    params: dict
    view: dict
    texts: dict[str, str]


def _calc_surr_opcost(
    coef: np.ndarray,
    feature_names: list[str],
    feature_cost_map: dict[str, OpCost],
) -> OpCost:
    active_mask = np.any(coef != 0, axis=0)
    active_features = [f for i, f in enumerate(feature_names) if active_mask[i]]
    nnz = np.count_nonzero(coef).item()
    surr_opcost = OpCost(mul=nnz, pm=max(0, nnz - int(coef.shape[0])))
    for feature in active_features:
        if feature not in feature_cost_map:
            raise ValueError(f"Found Unknown base func: '{feature}'")
        surr_opcost = surr_opcost + feature_cost_map[feature]
    return surr_opcost


def _calc_cost_stat(
    surr_opcost: OpCost, original_cost: OpCost | None
) -> dict[str, int]:
    if original_cost is None:
        return {}
    surr_d = surr_opcost.to_dict()
    orig_d = original_cost.to_dict()
    return {
        **{f"cost/surrogate/{k}": v for k, v in surr_d.items()},
        **{f"cost/original/{k}": v for k, v in orig_d.items()},
        **{f"cost/diff/{k}": surr_d[k] - orig_d[k] for k in orig_d},
    }


def sindy_analysis(
    neurosindy: "SINDyNeuroSurrogate",
    feature_cost_map: dict[str, OpCost],
    original_cost: OpCost | None = None,
) -> SINDySummary:
    result = SINDyResult(
        coef=neurosindy.sindy.coefficients(),
        feature_names=neurosindy.sindy.get_feature_names(),
        equations="\n".join(neurosindy.sindy.equations(precision=3)),
        source=neurosindy.source,
        target_names=neurosindy.target_names,
        params=neurosindy.sindy.optimizer.get_params(),
        preprocessor_metrics=calc_preprocessor_metrics(
            neurosindy.preprocessor, neurosindy.train_gate_data
        ),
    )
    surr_opcost = _calc_surr_opcost(result.coef, result.feature_names, feature_cost_map)
    nonzero_term_num = np.count_nonzero(result.coef).item()
    return SINDySummary(
        metrics={
            "nonzero_term_num": nonzero_term_num,
            "nonzero_term_ratio": float(nonzero_term_num / result.coef.size),
            **_calc_cost_stat(surr_opcost, original_cost),
            **result.preprocessor_metrics,
        },
        params=result.params,
        view={
            "xi_matrix": result.coef.tolist(),
            "feature_names": result.feature_names,
            "target_names": result.target_names,
        },
        texts={
            "equations.txt": result.equations,
            "coef.txt": np.array2string(result.coef, precision=3),
            "features.json": json.dumps(
                {k: v.to_dict() for k, v in feature_cost_map.items()}
            ),
            "misc/source.txt": result.source,
        },
    )
