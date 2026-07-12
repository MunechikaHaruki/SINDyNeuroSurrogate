import numpy as np

from ..registry.neurosindy import NeuroSurrogateBase
from .opcost import OpCost
from .result_bundle import PreprocessorBundle


def calc_preprocessor_metrics(preprocessor_bundle: PreprocessorBundle) -> dict:
    b = preprocessor_bundle.bundle
    if b is None:
        return {}
    return {
        "pca/explained_variance_ratio": float(b.explained_variance_ratio[0]),
        "pca/explained_variance": float(b.explained_variance[0]),
        "pca/reconstruction_mse": b.reconstruction_mse,
        "pca/reconstruction_mse_ratio": b.reconstruction_mse_ratio,
    }


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


def eval_surrogate(surrogate: NeuroSurrogateBase) -> dict:
    return {
        **calc_xi_metrics(surrogate.sindy_bundle.xi),
        **calc_preprocessor_metrics(surrogate.preprocessor_bundle),
        **calc_cost_stat(surrogate.opcost, surrogate.original_opcost),
    }
