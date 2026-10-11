"""シミュレーション Dataset のスキーマ read 規約を集約する access 層。

Dataset の座標規約 (features=MultiIndex(comp_id, variable, gate) /
gate=False→電位 / node_id→I_internal) を書くのが coords.py、読むのが本
モジュール。metrics・surrogate は生 sel を持たず本モジュール経由でのみ
Dataset を掘る。
"""

from __future__ import annotations

import numpy as np
import xarray as xr

POTENTIAL_VAR = "V"


def latent_vars(n: int) -> list[str]:
    """潜在ゲート変数名 z1..zN。preprocessed/surr Dataset の座標名・SINDy の列名・
    数式や図の表示名はすべてこの規約 1 本を共有する (roles.g の g と同じ意味)。"""
    return [f"z{i + 1}" for i in range(n)]


# --- 座標列挙 ---------------------------------------------------------------


def time(ds: xr.Dataset) -> np.ndarray:
    return ds["time"].values


def dt(ds: xr.Dataset) -> float:
    return float(ds["time"][1] - ds["time"][0])


def comp_ids(ds: xr.Dataset) -> np.ndarray:
    return np.unique(ds.coords["comp_id"].values)


# --- numpy accessor (計算層) ------------------------------------------------


def _variable_values(ds: xr.Dataset, comp_id: int, variable: str) -> np.ndarray:
    """comp_id・variable 名の系列値 (gate は variable 名で一意に決まる)。"""
    return ds["vars"].sel(comp_id=comp_id, variable=variable).to_numpy().squeeze()


def potential(ds: xr.Dataset, comp_id: int) -> np.ndarray:
    """comp_id の電位 V (gate=False) 系列値。"""
    return ds["vars"].sel(gate=False, comp_id=comp_id).to_numpy().squeeze()


def gate_matrix(ds: xr.Dataset, comp_id: int) -> np.ndarray:
    """comp_id のゲート/状態変数行列 (time, n_gate)。"""
    return ds["vars"].sel(gate=True, comp_id=comp_id).to_numpy()


def gate_matrices(ds: xr.Dataset, comp_ids: list[int]) -> list[np.ndarray]:
    """comp_ids 順のゲート/状態変数行列。"""
    return [gate_matrix(ds, comp_id) for comp_id in comp_ids]


def potentials(ds: xr.Dataset, comp_ids: list[int]) -> list[np.ndarray]:
    """comp_ids 順の電位系列。"""
    return [potential(ds, comp_id) for comp_id in comp_ids]


def comp_matrix(ds: xr.Dataset, comp_id: int) -> np.ndarray:
    """comp_id の全変数行列 (time, n_var)。"""
    return ds["vars"].sel(comp_id=comp_id).to_numpy()


def potential_matrix(ds: xr.Dataset) -> np.ndarray:
    """全 comp の電位行列 (time, N)。comp_id 昇順。"""
    return ds["vars"].sel(gate=False).sortby("comp_id").to_numpy()


def i_ext_values(ds: xr.Dataset) -> np.ndarray:
    return ds["I_ext"].to_numpy()


def i_internal_values(ds: xr.Dataset, comp_id: int) -> np.ndarray:
    return ds["I_internal"].sel(node_id=comp_id).to_numpy()
