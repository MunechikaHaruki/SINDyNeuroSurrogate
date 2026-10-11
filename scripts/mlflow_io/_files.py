"""冊が読む形で MLflow の artifact を書く (読むのは評価のキャッシュだけ)。

    <root>/meta.json           t0, dt, 点ごとの長さ n, features [[comp_id, 変数, gate]],
                               I_internal の nodes, comp 名 → id (渡したときだけ)
    <root>/<点>/comp<id>.f32   その comp の変数 (features の並び順)。変数ごとに連続
    <root>/<点>/I_ext.f32
    <root>/<点>/I_internal.f32 node ごとに連続

`.f32` は little-endian の float32 を並べただけのもの。JSON の nan と inf は null。
"""

import json
import math
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def write_f32(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(array, dtype="<f4").tobytes())


def _read_f32(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4")


def _comp_columns(features: Sequence[Sequence[Any]], comp_id: int) -> list[int]:
    """features の中で comp `comp_id` の列 (並び順のまま)。"""
    return [i for i, (comp, _, _) in enumerate(features) if comp == comp_id]


def write_waves(
    root: Path, waves: list[xr.Dataset], names: dict[str, int] | None = None
) -> None:
    """点の並びの波形を `root` へ書く。全点が同じ features を持つ。"""
    first = waves[0]
    features = [(int(c), str(v), bool(g)) for c, v, g in first.indexes["features"]]
    nodes = [int(i) for i in first["node_id"].values] if "I_internal" in first else None
    time = first["time"].values
    meta = {
        "t0": float(time[0]),
        "dt": float(time[1] - time[0]),
        "n": [ds.sizes["time"] for ds in waves],
        "features": features,
        "nodes": nodes,
        "names": names,
    }
    root.mkdir(parents=True)
    (root / "meta.json").write_text(json.dumps(meta))
    for index, ds in enumerate(waves):
        point = root / str(index)
        point.mkdir()
        values = ds["vars"].to_numpy()
        for comp in sorted({c for c, _, _ in features}):
            columns = _comp_columns(features, comp)
            write_f32(point / f"comp{comp}.f32", values[:, columns].T)
        write_f32(point / "I_ext.f32", ds["I_ext"].to_numpy())
        if nodes is not None:
            write_f32(point / "I_internal.f32", ds["I_internal"].to_numpy().T)


def read_waves(root: Path) -> list[xr.Dataset]:
    """`write_waves` の逆。"""
    meta = json.loads((root / "meta.json").read_text())
    features, nodes = meta["features"], meta["nodes"]
    index = pd.MultiIndex.from_tuples(
        [tuple(f) for f in features], names=("comp_id", "variable", "gate")
    )
    waves = []
    for point_index, n in enumerate(meta["n"]):
        point = root / str(point_index)
        values = np.empty((n, len(features)), dtype=np.float32)
        for comp in sorted({c for c, _, _ in features}):
            columns = _comp_columns(features, comp)
            raw = _read_f32(point / f"comp{comp}.f32")
            values[:, columns] = raw.reshape(len(columns), n).T
        data: dict[str, Any] = {
            "vars": (("time", "features"), values),
            "I_ext": (("time",), _read_f32(point / "I_ext.f32")),
        }
        coords: dict[Hashable, Any] = {
            "time": meta["t0"] + np.arange(n) * meta["dt"],
            **xr.Coordinates.from_pandas_multiindex(index, "features"),
        }
        if nodes is not None:
            raw = _read_f32(point / "I_internal.f32")
            data["I_internal"] = (("time", "node_id"), raw.reshape(len(nodes), n).T)
            coords["node_id"] = nodes
        waves.append(xr.Dataset(data, coords=coords))
    return waves


def _jsonable(value: Any) -> Any:
    """JSON に書ける素の値へ。nan と inf は null にする (JSON に無い値なので)。"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False))
