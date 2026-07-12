from ..registry.neurosindy import NeuroSurrogateBase
from .opcost import OpCost


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


def eval_surrogate(surrogate: NeuroSurrogateBase) -> dict:
    return {
        **surrogate.sindy_bundle.xi_metrics(),
        **surrogate.preprocessor_bundle.metrics(),
        **calc_cost_stat(surrogate.opcost, surrogate.original_opcost),
    }
