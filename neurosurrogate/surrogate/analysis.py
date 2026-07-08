from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.decomposition import PCA

from ..core.network import DatasetConfig
from ..core.simulator import unified_simulator
from ..opcost import OpCost
from ..registry.neuron import MCMODELS
from .libraries import FeatureLibrary
from .neurosindy import get_gate_numpy


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


def calc_surr_opcost(
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


def calc_cost_stat(surr_opcost: OpCost, original_cost: OpCost | None) -> dict[str, int]:
    if original_cost is None:
        return {}
    surr_d = surr_opcost.to_dict()
    orig_d = original_cost.to_dict()
    return {
        **{f"cost/surrogate/{k}": v for k, v in surr_d.items()},
        **{f"cost/original/{k}": v for k, v in orig_d.items()},
        **{f"cost/diff/{k}": surr_d[k] - orig_d[k] for k in orig_d},
    }


class SurrogateBundle(Protocol):
    xi: np.ndarray
    preprocessor: Any
    feature_names: list[str]
    target_names: list[str]
    library_specs: list[dict]
    dataset: DatasetConfig
    train_comp_id: int


def eval_surrogate(bundle: SurrogateBundle) -> dict:
    net = MCMODELS[bundle.dataset.model_name]
    train_gate = get_gate_numpy(unified_simulator(bundle.dataset), bundle.train_comp_id)
    feature_cost_map = FeatureLibrary.build(bundle.library_specs).to_base_cost(
        bundle.target_names + ["u"]
    )
    surr_opcost = calc_surr_opcost(bundle.xi, bundle.feature_names, feature_cost_map)
    original_cost = net.nodes[bundle.train_comp_id].type.opcost
    nnz = int((bundle.xi != 0).sum())
    return {
        "nnz": nnz,
        "nnz_ratio": nnz / bundle.xi.size,
        **calc_preprocessor_metrics(bundle.preprocessor, train_gate),
        **calc_cost_stat(surr_opcost, original_cost),
    }
