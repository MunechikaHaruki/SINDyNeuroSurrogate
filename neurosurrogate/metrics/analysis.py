import numpy as np
from sklearn.decomposition import PCA

from ..core.network import DatasetConfig
from ..core.simulator import unified_simulator
from ..registry.neurosindy import NeuroSurrogateBase, get_gate_numpy
from .opcost import OpCost


def calc_preprocessor_metrics(preprocessor, dataset_cfg: DatasetConfig, train_comp_id: int) -> dict:
    train_gate = get_gate_numpy(unified_simulator(dataset_cfg), train_comp_id)

    def _get_pca_metrics(pca: PCA) -> dict:
        reconstructed = pca.inverse_transform(pca.transform(train_gate))
        mse = np.mean((train_gate - reconstructed) ** 2)
        return {
            "pca/explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "pca/explained_variance": float(pca.explained_variance_[0]),
            "pca/reconstruction_mse": float(mse),
            "pca/reconstruction_mse_ratio": float(mse / np.var(train_gate)),
        }

    if isinstance(preprocessor, PCA):
        return _get_pca_metrics(preprocessor)
    return {}


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


def calc_xi_metrics(xi: np.ndarray) -> dict[str, float]:
    nnz = int((xi != 0).sum())
    return {"nnz": nnz, "nnz_ratio": nnz / xi.size}


def eval_surrogate(surrogate: NeuroSurrogateBase, dataset_cfg: DatasetConfig) -> dict:
    return {
        **calc_xi_metrics(surrogate.xi),
        **calc_preprocessor_metrics(surrogate.preprocessor, dataset_cfg, surrogate.train_comp_id),
        **calc_cost_stat(surrogate.opcost, surrogate.original_opcost),
    }
