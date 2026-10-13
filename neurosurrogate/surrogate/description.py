"""学習済み surrogate が自分について言えることを、図に依らないデータで出す。

何を学習したか (閉包項と前処理の中身) と、何を食わせたか (学習データと、そこから
同定器へ渡した列) を数値で組む。**図は描かない** — 描くのは figures/ の図で、
保存の形は呼ぶ側 (`scripts/mlflow_io`) が決める。置換シミュは回さないので、run を
読んだだけで出せる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import xarray as xr

from ..core import access
from .model import Surrogate


@dataclass(frozen=True)
class _CompFit:
    """学習 comp 1 つについて、閉包項の手前で起きたこと。時間軸は学習データと同じ。"""

    x: np.ndarray  # (time, len(x_names)) 同定器へ渡した状態列
    latent: np.ndarray  # (time, n_components) 学習ゲートを潜在へ落としたもの
    recon_rmse: np.ndarray  # (time,) ゲート → 潜在 → ゲートの RMSE


@dataclass(frozen=True)
class _Description:
    """`model` は JSON にできる dict、残りは配列。"""

    model: dict[str, Any]
    training_data: xr.Dataset
    fits: dict[int, _CompFit]  # 学習 comp id → その comp の _CompFit


def describe(surrogate: Surrogate) -> _Description:
    """surrogate 1 つの中身。学習データは spec から再現する (simulate が走る)。"""
    spec, data = surrogate.spec, surrogate.training_data
    net = spec.dataset.net
    n_gate = surrogate.ansatz.n_train_gate(spec)
    inputs = surrogate.ansatz.train_inputs(spec, data, surrogate.preprocessor)
    train_ids = spec.train_comp_ids()
    fits = {}
    for pos, comp_id in enumerate(train_ids):
        gate = access.gate_matrix(data, comp_id)[:, :n_gate]
        latent = np.asarray(surrogate.preprocessor.encode(gate))
        recon = np.asarray(surrogate.preprocessor.decode(jnp.asarray(latent)))
        fits[comp_id] = _CompFit(
            x=np.asarray(inputs.x[pos]),
            latent=latent,
            recon_rmse=np.sqrt(np.mean((gate - recon) ** 2, axis=1)),
        )
    model = {
        "closure": surrogate.closure.describe(),
        "preprocessor": surrogate.preprocessor.describe(),
        "net": {
            "nodes": [{"name": c.name, "type": c.type.name} for c in net.nodes],
            "edges": [
                {"src": e.src, "dst": e.dst, "weight": e.weight} for e in net.edges
            ],
            "stim": spec.dataset.stim,
            # 置換してよいノード (学習ドメイン) と、学習軌道を取ったノード。
            "replaceable": [c.name for c in net.nodes if spec.in_train_domain(c)],
            "train": [net.nodes[i].name for i in train_ids],
        },
        "train_gates": list(spec.comp_type.gate_names[:n_gate]),
        "x_names": list(inputs.x_names),
        "latent_names": access.latent_vars(spec.n_components),
    }
    return _Description(model, data, fits)
