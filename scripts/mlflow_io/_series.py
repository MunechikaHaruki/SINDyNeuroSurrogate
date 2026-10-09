"""波形 experiment (`_EVAL_EXP`): **1 run = 1 `sim.result.SeriesRun`** = 1 列の波形。
kind=original (原系、`series_hash` で共有) と kind=surrogate が平坦に並び、置換系は
`original_hash` で原系を名指す。点は run の中の並び順そのもの。

波形は live-textbook の冊も読むので、TS で読める形で置く (`_write_waves`)。置換系の
run には、原系と比べた指標 (`metrics.json`)、原系を潜在空間へ射影した軌道
(`projected/<comp>/`)、学習側の指標 (`summary.json`) も置く = 図に要るものは全部
run の中にあり、どの列を並べて見るかは読む側が選ぶ。

    waves/meta.json            t0, dt, 点ごとの長さ n, features [[comp_id, 変数, gate]],
                               I_internal の nodes, comp 名 → id (原系だけ)
    waves/<点>/comp<id>.f32    その comp の変数 (features の並び順)。変数ごとに連続
    waves/<点>/I_ext.f32
    waves/<点>/I_internal.f32  node ごとに連続

`.f32` は little-endian の float32 を並べただけのもの。
"""

import json
import math
import os
import tempfile
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import xarray as xr

from neurosurrogate.core.coords import transform_gate
from neurosurrogate.sim.result import SeriesResults, SeriesRun
from neurosurrogate.sim.run import run_column
from neurosurrogate.sim.spec import EvalSeries
from neurosurrogate.sim.waveform import compare
from neurosurrogate.surrogate.model import Surrogate

from . import logger
from ._query import exp_id, latest_by_tag

_EVAL_EXP = os.environ.get("MLFLOW_EVAL_EXPERIMENT", "eval_series")
_WAVES_DIR = "waves"
_FORMAT = "f32"  # 保存形式の版。鍵に混ぜる = 形式を変えたら古い run は鍵が合わない
_HASH_TAG = "series_hash"  # 同一性 = 掃引 (+ 置換系なら学習 run_id) + 形式
_KIND_ORIGINAL = "original"
_KIND_SURROGATE = "surrogate"


def _series_hash(series: EvalSeries, run_id: str | None) -> str:
    """「この掃引をこの surrogate で既に回したか」の鍵。原系は `hash` (置換範囲を
    含まない = 置換範囲だけが違う対照系列と共有される)、置換系は `replaced_hash`
    (置換範囲を含む) に学習 run_id を足す。"""
    base = series.hash() if run_id is None else f"{series.replaced_hash()}-{run_id}"
    return f"{base}-{_FORMAT}"


def _tags(name: str, series: EvalSeries, run_id: str | None) -> dict[str, str]:
    """波形列の同一性・種類・系列名・原系への参照を MLflow tag へ落とす。"""
    return {
        _HASH_TAG: _series_hash(series, run_id),
        "kind": _KIND_ORIGINAL if run_id is None else _KIND_SURROGATE,
        "series_name": name,
        **({} if run_id is None else {"original_hash": _series_hash(series, None)}),
    }


# --- 保存形式 ------------------------------------------------------------------


def _write_f32(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(array, dtype="<f4").tobytes())


def _read_f32(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4")


def _comp_columns(features: Sequence[Sequence[Any]], comp_id: int) -> list[int]:
    """features の中で comp `comp_id` の列 (並び順のまま)。"""
    return [i for i, (comp, _, _) in enumerate(features) if comp == comp_id]


def _write_waves(
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
            _write_f32(point / f"comp{comp}.f32", values[:, columns].T)
        _write_f32(point / "I_ext.f32", ds["I_ext"].to_numpy())
        if nodes is not None:
            _write_f32(point / "I_internal.f32", ds["I_internal"].to_numpy().T)


def _read_waves(root: Path) -> list[xr.Dataset]:
    """`_write_waves` の逆。"""
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False))


def _write_comparison(root: Path, view: SeriesResults, surrogate: Surrogate) -> None:
    """置換系 1 列を原系と比べたものを書く。指標は全 comp・全点の分 (どの comp を
    見るかは読む側が選ぶ)、射影は置換した comp の分。"""
    series, column = view.series, view.surrs[0]
    net, dt = series.spec.net, series.spec.dt
    points = range(len(view.original_waves))
    metrics = {
        comp: [
            compare(*view.pair(index, column), net.name_to_idx(comp), dt)
            for index in points
        ]
        for comp in net.names
    }
    _write_json(root / "metrics.json", metrics)
    _write_json(root / "summary.json", surrogate.summary())
    for comp in series.replace_targets:
        comp_id = net.name_to_idx(comp)
        _write_waves(
            root / "projected" / comp,
            [
                transform_gate(surrogate.preprocessor, ds, comp_id)
                for ds in view.original_waves
            ],
        )


# --- run の探索と保存 -------------------------------------------------------------


def run_series(
    name: str,
    series: EvalSeries,
    run_id: str | None = None,
    surrogate: Surrogate | None = None,
) -> str:
    """1 列 → 波形 run の id。同じ掃引を同じ surrogate で回した run があればそれを返す
    = **回さない** (シミュは決定的なので、鍵が一致した run は常に正しい)。
    `run_id`/`surrogate` が両方 `None` なら原系。置換系は原系の run も用意して比べる。
    """
    found = latest_by_tag(_EVAL_EXP, _HASH_TAG, _series_hash(series, run_id))
    if found is not None:
        return str(found.info.run_id)

    column = run_column(series, run_id, surrogate)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        if surrogate is None:
            names = {
                comp: series.spec.net.name_to_idx(comp)
                for comp in series.spec.net.names
            }
            _write_waves(root / _WAVES_DIR, column.waves, names)
        else:
            _write_waves(root / _WAVES_DIR, column.waves)
            original = _load_column(run_series(name, series))
            _write_comparison(root, SeriesResults(original, (column,)), surrogate)
        kind = _KIND_ORIGINAL if run_id is None else f"{_KIND_SURROGATE}:{run_id[:8]}"
        with mlflow.start_run(
            experiment_id=exp_id(_EVAL_EXP), run_name=f"{name} [{kind}]"
        ) as run:
            mlflow.log_params(
                {
                    "series": json.dumps(series.to_dict(), sort_keys=True, default=str),
                    # MLflow の param は文字列 → None は "None" と書かれて読み戻しで
                    # 区別できない。空文字を「無し」の綴りに統一する。
                    "run_id": run_id or "",
                    "unit": series.unit,
                }
            )
            mlflow.set_tags(_tags(name, series, run_id))
            mlflow.log_artifacts(tmp)
    logger.info("評価 run 保存: %s [%s] (%s)", name, kind, run.info.run_id)
    return str(run.info.run_id)


def _load_column(eval_run_id: str) -> SeriesRun:
    """波形 run の id → **その run が保存した列そのもの** (`run_series` の逆)。記述も
    run_id も param が持つので、呼ぶ側は id 1 つを指すだけでよい。"""
    params = mlflow.get_run(eval_run_id).data.params
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            f"runs:/{eval_run_id}/{_WAVES_DIR}", dst_path=tmp
        )
        return SeriesRun(
            EvalSeries.from_dict(json.loads(params["series"])),
            # 空文字が「無し」の綴り (MLflow の param は文字列なので None を書けない)。
            params["run_id"] or None,
            _read_waves(Path(local)),
        )
