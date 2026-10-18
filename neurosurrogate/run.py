"""計算の入口: `uv run python -m neurosurrogate.run <spec.json> <runs/ハッシュ>`。

spec.json 1 つを受け取り、run のディレクトリを埋めるだけの純粋な計算。何を回すか、
掃引、ハッシュ、重複の回避、run の一覧は bun の段取り
(`evaluation/orchestrate.ts`) が持ち、spec の形は同じ
場所の `conf.ts` が持つ。ドメインのどの層からも import されない最上位の層。
spec の種類は 3 つ (`kind`):

- `train`    学習。`surrogate` は設定ツリー (spec + preprocessor/ansatz/closure)
- `original` 原系の掃引。`series` は `EvalSeries.to_dict` から置換範囲を除いたもの
- `replaced` 置換系の掃引。`series` に置換範囲が入り、`train` と `original` が
             同じ runs/ の下の学習 run と原系 run のハッシュを指す

書くもの (図が読む形。`.f32` は little-endian の float32 を並べただけ、
JSON の nan は null):

    spec.json  meta.json (kind, name, created, 掃引の単位)
    train      surrogate/ (spec.json + pickle)、description/model.json、
               description/train/ (waves と同じ形、点は 1 つ)、
               description/fit/<comp_id>/{x,latent,recon_rmse}.f32
    original   waves/
    replaced   waves/、metrics.json (comp 名 → 点ごとの compare)、summary.json、
               projected/<comp>/ (置換した comp の原系ゲートを潜在へ射影したもの。
               waves と同じ形)

    waves/meta.json           t0, dt, 点ごとの長さ n, features [[comp_id, 変数, gate]],
                              I_internal の nodes, comp 名 → id
    waves/<点>/comp<id>.f32   その comp の変数 (features の並び順)。変数ごとに連続
    waves/<点>/I_ext.f32      waves/<点>/I_internal.f32 (node ごとに連続)

途中で落ちても半端な run を残さないよう、隣の一時ディレクトリ (`<ハッシュ>.partial`)
に書いてから名前を変える。一時ディレクトリには書いているプロセスの pid を置き、図は
それで「計算中」を出す。ディレクトリが既にあれば回さない。置換できない組み合わせは何も書かずに
終了コード 3、同じ run を生きた別のプロセスが書いていれば終了コード 4。
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
from collections.abc import Hashable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .core.coords import transform_gate
from .sim.result import SeriesResults, SeriesRun
from .sim.run import run_column
from .sim.spec import EvalSeries
from .sim.waveform import compare
from .surrogate.description import describe
from .surrogate.model import Surrogate, fit_surrogate
from .surrogate.replace import applicable

_FORMAT = 1  # spec の形式の版。段取りが spec に入れる = 形式を変えたら鍵が変わる
_NOT_REPLACEABLE = 3
_RUNNING = 4
_logger = logging.getLogger(__name__)


# --- 波形の形式 ---------------------------------------------------------------


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
            _write_f32(
                point / f"comp{comp}.f32", values[:, _comp_columns(features, comp)].T
            )
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


# --- 3 種類の run ---------------------------------------------------------------


def _names(series: EvalSeries) -> dict[str, int]:
    net = series.spec.net
    return {comp: net.name_to_idx(comp) for comp in net.names}


def _train(root: Path, spec: dict) -> dict:
    """学習して surrogate とその中身 (`description/`) を書く。"""
    surrogate = fit_surrogate(spec["surrogate"])
    (root / "surrogate").mkdir()
    surrogate.save(root / "surrogate")
    desc = describe(surrogate)
    _write_json(root / "description/model.json", desc.model)
    net = surrogate.spec.dataset.net
    names = {comp: net.name_to_idx(comp) for comp in net.names}
    _write_waves(root / "description/train", [desc.training_data], names)
    for comp_id, fit in desc.fits.items():
        folder = root / "description/fit" / str(comp_id)
        folder.mkdir(parents=True)
        _write_f32(folder / "x.f32", fit.x.T)
        _write_f32(folder / "latent.f32", fit.latent.T)
        _write_f32(folder / "recon_rmse.f32", fit.recon_rmse)
    return {}


def _series(spec: dict) -> EvalSeries:
    return EvalSeries.from_dict({"replace_targets": [], **spec["series"]})


def _original(root: Path, spec: dict) -> dict:
    """原系の掃引を回して波形を書く。"""
    series = _series(spec)
    _write_waves(root / "waves", run_column(series, None, None).waves, _names(series))
    return {"unit": series.unit}


def _replaced(root: Path, spec: dict) -> dict:
    """置換系の掃引を回し、原系と比べた指標・射影・学習側の指標も書く。指標は全 comp・
    全点の分 (どの comp を見るかは図が選ぶ)、射影は置換した comp の分。"""
    series = _series(spec)
    surrogate = Surrogate.load(root.parent / spec["train"] / "surrogate")
    if not applicable(surrogate.spec, series):
        print(f"{spec['train']} は {series.replace_targets} を置換できない")
        sys.exit(_NOT_REPLACEABLE)
    original = SeriesRun(
        series, None, _read_waves(root.parent / spec["original"] / "waves")
    )
    column = run_column(series, spec["train"], surrogate)
    _write_waves(root / "waves", column.waves, _names(series))
    view = SeriesResults(original, (column,))
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
    return {"unit": series.unit}


_KINDS = {"train": _train, "original": _original, "replaced": _replaced}


def _writer_alive(partial: Path) -> bool:
    """書きかけの run を書いているプロセスが生きているか (落ちたあとの残骸なら偽)。"""
    try:
        os.kill(int((partial / "pid").read_text()), 0)
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _run(spec_path: Path, root: Path, name: str) -> None:
    """spec を回して `root` を埋める。`root` があれば回さない。"""
    if root.exists():
        _logger.info("%s は回してある", root.name)
        return
    spec = json.loads(spec_path.read_text())
    if spec.get("format") != _FORMAT:
        raise ValueError(f"spec の形式 {spec.get('format')} は {_FORMAT} でない")
    partial = root.with_name(f"{root.name}.partial")
    if _writer_alive(partial):
        print(f"{root.name} は別のプロセスが書いている")
        sys.exit(_RUNNING)
    shutil.rmtree(partial, ignore_errors=True)
    try:
        partial.mkdir(parents=True)
    except FileExistsError:
        sys.exit(_RUNNING)
    (partial / "pid").write_text(str(os.getpid()))
    try:
        (partial / "spec.json").write_text(spec_path.read_text())
        extra = _KINDS[spec["kind"]](partial, spec)
        meta = {
            "kind": spec["kind"],
            "name": name,
            "created": datetime.now().isoformat(timespec="seconds"),
            **extra,
        }
        _write_json(partial / "meta.json", meta)
        (partial / "pid").unlink()
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    partial.rename(root)
    _logger.info("%s (%s) を書いた", root.name, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("spec", type=Path, help="spec.json")
    parser.add_argument("root", type=Path, help="埋める run のディレクトリ")
    parser.add_argument("--name", default="", help="図に出す名前 (同一性に入らない)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _run(args.spec, args.root, args.name)


if __name__ == "__main__":
    main()
