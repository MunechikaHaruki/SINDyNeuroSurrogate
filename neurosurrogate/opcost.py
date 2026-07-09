from dataclasses import dataclass, fields

import numpy as np


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


def calc_surr_opcost(
    coef: np.ndarray,
    feature_names: list[str],
    feature_cost_map: dict[str, "OpCost"],
) -> "OpCost":
    active_mask = np.any(coef != 0, axis=0)
    active_features = [f for i, f in enumerate(feature_names) if active_mask[i]]
    nnz = np.count_nonzero(coef).item()
    surr_opcost = OpCost(mul=nnz, pm=max(0, nnz - int(coef.shape[0])))
    for feature in active_features:
        if feature not in feature_cost_map:
            raise ValueError(f"Found Unknown base func: '{feature}'")
        surr_opcost = surr_opcost + feature_cost_map[feature]
    return surr_opcost
