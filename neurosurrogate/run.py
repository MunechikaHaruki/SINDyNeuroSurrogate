"""計算の入口: `uv run python -m neurosurrogate.run <spec.json> <run のディレクトリ>`。

spec.json 1 つを受け取り、run のディレクトリを埋めるだけの純粋な計算。何を回すか、
掃引、ハッシュ、重複の回避、run の置き場と一覧は bun の段取り
(`evaluation/orchestrate/orchestrate.ts`) が持ち、spec の形は同じ
場所の `spec.ts` が持つ。ドメインのどの層からも import されない最上位の層。
spec の種類は 2 つ (`kind`):

- `train` 学習。`surrogate` は設定 (surrogate.Config: 区画の型、形、学習データ、群)。
          置き場は `runs/models/<ハッシュ>/`
- `sim`   入力 × モデルのシミュレーション。`input` は `EvalSeries.to_dict`
          から置換範囲を除いたもの (電流と掃引)。`model` は
          `{train, replace_targets}` か null (元のモデル)。置き場は
          `runs/inputs/<入力のハッシュ>/` の下で、元のモデルは `original/`、
          置き換えたモデルは段取りが組んだ鍵の名前。`train` は
          `runs/models/` の学習 run を指す

書くもの (図が読む形。`.f64` は little-endian の float64 を並べただけ、
JSON の nan は null):

    spec.json  meta.json (kind = train / original / replaced, name, created, 掃引の単位)
    train      surrogate/ (config.json + parts.joblib)、summary.json (学習側の指標)、
               description/model.json、description/train/ (waves と同じ形、点は 1 つ)、
               description/fit/<comp_id>/{x,latent,recon_rmse}.f64
    sim        waves/。置き換えたモデルは加えて、metrics.json (comp 名 →
               点ごとの、同じ入力の original/ との compare)、projected/<comp>/
               (置換した comp の original/ のゲートを潜在へ射影したもの。
               waves と同じ形)

    waves/meta.json           t0, dt, 点ごとの長さ n, features [[comp_id, 変数, gate]],
                              I_internal の nodes, comp 名 → id
    waves/<点>/comp<id>.f64   その comp の変数 (features の並び順)。変数ごとに連続
    waves/<点>/I_ext.f64      waves/<点>/I_internal.f64 (node ごとに連続)

途中で落ちても半端な run を残さないよう、隣の一時ディレクトリ (`<名前>.partial`)
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

from .metrics import compare, diverged
from .physics.simulate import (
    EvalSeries,
    gate_matrix,
    i_internal,
    latent_dataset,
    potential,
    simulate,
)
from .surrogate.model import (
    Surrogate,
    applicable,
    describe,
    fit,
    replace,
    summary,
)

_FORMAT = 5  # spec の形式の版。段取りが spec に入れる = 形式を変えたら鍵が変わる
_NOT_REPLACEABLE = 3
_RUNNING = 4
_logger = logging.getLogger(__name__)


# --- 波形の形式 ---------------------------------------------------------------


def _write_f64(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(array, dtype="<f8").tobytes())


def _read_f64(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f8")


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
            _write_f64(
                point / f"comp{comp}.f64", values[:, _comp_columns(features, comp)].T
            )
        _write_f64(point / "I_ext.f64", ds["I_ext"].to_numpy())
        if nodes is not None:
            _write_f64(point / "I_internal.f64", ds["I_internal"].to_numpy().T)


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
        values = np.empty((n, len(features)), dtype=np.float64)
        for comp in sorted({c for c, _, _ in features}):
            columns = _comp_columns(features, comp)
            raw = _read_f64(point / f"comp{comp}.f64")
            values[:, columns] = raw.reshape(len(columns), n).T
        data: dict[str, Any] = {
            "vars": (("time", "features"), values),
            "I_ext": (("time",), _read_f64(point / "I_ext.f64")),
        }
        coords: dict[Hashable, Any] = {
            "time": meta["t0"] + np.arange(n) * meta["dt"],
            **xr.Coordinates.from_pandas_multiindex(index, "features"),
        }
        if nodes is not None:
            raw = _read_f64(point / "I_internal.f64")
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


# --- 2 種類の run ---------------------------------------------------------------


def _names(series: EvalSeries) -> dict[str, int]:
    net = series.spec.net
    return {comp: net.index(comp) for comp in net.names}


def _train(root: Path, spec: dict) -> dict:
    """学習して surrogate とその中身 (`description/`) を書く。"""
    surrogate = fit(spec["surrogate"])
    (root / "surrogate").mkdir()
    surrogate.save(root / "surrogate")
    model, fits = describe(surrogate)
    _write_json(root / "description/model.json", model)
    net = surrogate.config.data.net
    names = {comp: net.index(comp) for comp in net.names}
    _write_waves(root / "description/train", [surrogate.training_data], names)
    for comp_id, comp_fit in fits.items():
        folder = root / "description/fit" / str(comp_id)
        folder.mkdir(parents=True)
        _write_f64(folder / "x.f64", comp_fit.x.T)
        _write_f64(folder / "latent.f64", comp_fit.latent.T)
        _write_f64(folder / "recon_rmse.f64", comp_fit.recon_rmse)
    _write_json(root / "summary.json", summary(surrogate))
    return {}


def _project(surrogate: Surrogate, ds: xr.Dataset, comp_id: int) -> xr.Dataset:
    """元のモデルの区画のゲートを、各群の圧縮器で潜在へ落とした [V, z...]。"""
    config = surrogate.config
    latents = []
    for (compressor, _), group in zip(surrogate.parts, config.groups, strict=True):
        gates = gate_matrix(ds, comp_id)[:, config.positions(group)]
        latents.append(compressor.encode(gates))
    return latent_dataset(
        potential(ds, comp_id),
        np.concatenate(latents, axis=1),
        i_internal(ds, comp_id),
        comp_id,
        float(ds["time"][1] - ds["time"][0]),
    )


def _sim(root: Path, spec: dict) -> dict:
    """入力をモデルで回して波形を書く。置き換えたモデルなら、同じ入力の元のモデル
    (`original/`) と比べた指標 (全区画・全点) と、置換した区画の射影も書く。"""
    model = spec["model"]
    targets = model["replace_targets"] if model else []
    series = EvalSeries.from_dict({**spec["input"], "replace_targets": targets})
    if model is None:
        waves = [simulate(point.run_input()) for point in series.points]
        _write_waves(root / "waves", waves, _names(series))
        return {"unit": series.unit}
    runs = root.parents[2]
    surrogate = Surrogate.load(runs / "models" / model["train"] / "surrogate")
    if not applicable(surrogate.config, series):
        print(f"{model['train']} は {targets} を置換できない")
        sys.exit(_NOT_REPLACEABLE)
    originals = _read_waves(root.parent / "original/waves")
    waves = []
    for point in series.points:
        ds = simulate(replace(surrogate, point.run_input(), targets))
        if broken := [
            c for c in targets if diverged(potential(ds, point.net.index(c)))
        ]:
            _logger.warning("置換系の電位が発散 (%s): %s", point.current_type, broken)
        waves.append(ds)
    _write_waves(root / "waves", waves, _names(series))
    net, dt = series.spec.net, series.spec.dt
    metrics = {}
    for comp in net.names:
        metrics[comp] = [
            compare(originals[i], waves[i], net.index(comp), dt)
            for i in range(len(originals))
        ]
    _write_json(root / "metrics.json", metrics)
    for comp in targets:
        projected = [_project(surrogate, ds, net.index(comp)) for ds in originals]
        _write_waves(root / "projected" / comp, projected)
    return {"unit": series.unit}


def _kind(spec: dict) -> str:
    """図に出す run の種類。sim はモデルの有無で、元のモデルと置き換えたモデルに
    分ける。"""
    if spec["kind"] == "train":
        return "train"
    return "original" if spec["model"] is None else "replaced"


_KINDS = {"train": _train, "sim": _sim}


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
            "kind": _kind(spec),
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
