"""SINDy 同定の共有 leaf: TrainInputs → SINDyBundle。

closure/sindy は leaf (ansatz の TrainInputs を知らない) → `from_sindy` は素データで
受ける。その「TrainInputs を素データへ開いて渡す」変換をここへ集約し、SINDyAnsatz /
HybridAnsatz の fit が roles だけ差し替えて共有する (列構造 = roles が定式化ごとの唯一
の違い)。
"""

import numpy as np

from ...closure.sindy import SINDyBundle
from ...closure.sindy.roles import Roles
from ..base import TrainInputs


def fit_sindy(
    inputs: TrainInputs, t: np.ndarray, roles: Roles, spec: dict
) -> SINDyBundle:
    """列名を Symbol へ、軌道をそのまま (comp ごとに分割) 同定器へ流す。"""
    return SINDyBundle.from_sindy(
        library_specs=spec["library_specs"],
        optimizer_spec=spec["optimizer"],
        x=inputs.x,
        u=inputs.u,
        t=[t] * len(inputs.x),
        targets=inputs.target_symbols(),
        inputs=inputs.input_symbols(),
        roles=roles,
    )
