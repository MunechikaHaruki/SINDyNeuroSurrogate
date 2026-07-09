from dataclasses import dataclass, fields

import numpy as np

from .sindy_result import SINDyResult


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


def calc_sindy_opcost(
    result: SINDyResult,
    feature_cost_map: dict[str, "OpCost"],
) -> "OpCost":
    active_mask = np.any(result.xi != 0, axis=0)
    active_features = [f for i, f in enumerate(result.feature_names) if active_mask[i]]
    nnz = np.count_nonzero(result.xi).item()
    surr_opcost = OpCost(mul=nnz, pm=max(0, nnz - int(result.xi.shape[0])))
    for feature in active_features:
        if feature not in feature_cost_map:
            raise ValueError(f"Found Unknown base func: '{feature}'")
        surr_opcost = surr_opcost + feature_cost_map[feature]
    return surr_opcost
