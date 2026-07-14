"""シミュレーション Dataset のスキーマ read 規約を集約する access 層。

Dataset の座標規約 (features=MultiIndex(comp_id, variable, gate) /
gate=False→電位 / node_id→I_internal) を書くのが coords.py、読むのが本
モジュール。描画・metrics・surrogate は生 sel を持たず本モジュール経由で
のみ Dataset を掘り、純粋な (t, y) / numpy で扱う。
"""

from __future__ import annotations

import numpy as np
import xarray as xr

Trace = tuple[np.ndarray, np.ndarray]  # (time, value)

POTENTIAL_VAR = "V"


def _trace(da: xr.DataArray) -> Trace:
    return da["time"].values, da.values.squeeze()


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


def trace(ds: xr.Dataset, comp_id: int, variable: str) -> Trace:
    """comp_id・variable 名で系列を取得 (gate は variable 名で一意に決まる)。"""
    return _trace(ds["vars"].sel(comp_id=comp_id, variable=variable))


def i_ext(ds: xr.Dataset) -> Trace:
    return _trace(ds["I_ext"])


def has_i_internal(ds: xr.Dataset) -> bool:
    return "I_internal" in ds


def i_internal(ds: xr.Dataset, comp_id: int) -> Trace:
    return _trace(ds["I_internal"].sel(node_id=comp_id))
