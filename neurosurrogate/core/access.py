"""シミュレーション Dataset のスキーマ read 規約を集約する access 層。

Dataset の座標規約 (features=MultiIndex(comp_id, variable, gate) /
gate=False→電位 / node_id→I_internal) を書くのが coords.py、読むのが本
モジュール。描画・metrics・surrogate は生 sel を持たず本モジュール経由で
のみ Dataset を掘る。値のみ要る計算層は numpy accessor、時間軸を要る描画層
は (t, y) accessor を使う。
"""

from __future__ import annotations

import numpy as np
import xarray as xr

Trace = tuple[np.ndarray, np.ndarray]  # (time, value)

POTENTIAL_VAR = "V"


# --- 座標列挙 ---------------------------------------------------------------


def time(ds: xr.Dataset) -> np.ndarray:
    return ds["time"].values


def dt(ds: xr.Dataset) -> float:
    return float(ds["time"][1] - ds["time"][0])


def comp_ids(ds: xr.Dataset) -> np.ndarray:
    return np.unique(ds.coords["comp_id"].values)


def gate_variables(ds: xr.Dataset, comp_id: int) -> np.ndarray:
    """comp_id のゲート/状態変数 (gate=True) 名を列挙。gate 無し comp は空。"""
    sub = ds["vars"].sel(comp_id=comp_id)
    mask = sub.coords["gate"].values
    return np.unique(sub.coords["variable"].values[mask])


def latent_variables(ds: xr.Dataset) -> list[str]:
    """電位以外の変数名 (latent 系) を列挙。"""
    return [v for v in ds.coords["variable"].values if v != POTENTIAL_VAR]


# --- numpy accessor (計算層) ------------------------------------------------


def variable_values(ds: xr.Dataset, comp_id: int, variable: str) -> np.ndarray:
    """comp_id・variable 名の系列値 (gate は variable 名で一意に決まる)。"""
    return ds["vars"].sel(comp_id=comp_id, variable=variable).to_numpy().squeeze()


def potential(ds: xr.Dataset, comp_id: int) -> np.ndarray:
    """comp_id の電位 V (gate=False) 系列値。"""
    return ds["vars"].sel(gate=False, comp_id=comp_id).to_numpy().squeeze()


def gate_matrix(ds: xr.Dataset, comp_id: int) -> np.ndarray:
    """comp_id のゲート/状態変数行列 (time, n_gate)。"""
    return ds["vars"].sel(gate=True, comp_id=comp_id).to_numpy()


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


def has_i_internal(ds: xr.Dataset) -> bool:
    return "I_internal" in ds


# --- (t, y) accessor (描画層) -----------------------------------------------


def trace(ds: xr.Dataset, comp_id: int, variable: str) -> Trace:
    return time(ds), variable_values(ds, comp_id, variable)


def i_ext(ds: xr.Dataset) -> Trace:
    return time(ds), i_ext_values(ds)


def i_internal(ds: xr.Dataset, comp_id: int) -> Trace:
    return time(ds), i_internal_values(ds, comp_id)
