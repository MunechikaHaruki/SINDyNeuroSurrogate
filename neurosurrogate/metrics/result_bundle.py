from dataclasses import dataclass

import numpy as np
import pysindy as ps


@dataclass
class SINDyBundle:
    xi: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    equations: str
    library_specs: list[dict]

    @classmethod
    def from_sindy(
        cls, sindy: ps.SINDy, target_names: list[str], library_specs: list[dict]
    ) -> "SINDyBundle":
        return cls(
            xi=sindy.coefficients(),
            feature_names=sindy.get_feature_names(),
            target_names=target_names,
            equations="\n".join(sindy.equations(precision=3)),
            library_specs=library_specs,
        )
