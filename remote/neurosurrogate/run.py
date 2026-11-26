"""計算の入口: `uv run python -m neurosurrogate.run <spec.json> <run のディレクトリ>`。

spec.json 1 つを受け取り、run のディレクトリを埋めるだけの純粋な計算。掃引は手元が
入力 1 本ずつに開いて送り、何を回すか、ハッシュ、重複の回避、run の置き場と一覧は
bun の段取り (`../run/`) が持ち、spec の形は
`../../shared/spec.ts` が持つ。ドメインのどの層からも import されない最上位の層。
spec の種類は 4 つ (`kind`)。学習の入力の元のモデル → 学習器 → モデル →
シミュレーションの順に、後ろの spec が前の run を鍵で指す。学習データは置き場を
持たず、学習の入力 (学習用の電流の単発の入力) の元のモデル
`runs/inputs/<入力の鍵>/original/` がそれにあたる:

- `part`  学習器 1 つ (群 1 つの圧縮器と右辺)。`input` は学習の入力の鍵で、
          `cell`・`form`・`train_comp`・`group` で学習する。
          置き場は `runs/parts/<ハッシュ>/`
- `model` 学習器の並びを組んだモデル。`parts` は学習器のハッシュの並び (群の順)、
          `cell`・`form`・`absorbed`。学習はせず、組んだ演算コストを書く。KAM の
          モデルは `parts` が空で、`kam` に群と学習の入力の鍵と区画を持つ。
          置き場は `runs/models/<ハッシュ>/`
- `sim`   入力 × モデルのシミュレーション。`input` は電流 1 本 (`{spec}`。掃引は
          手元が点ごとの入力に開いて送る)。`model` は
          `{surrogate, replace_targets}` か null (元のモデル)。置き場は
          `runs/inputs/<入力のハッシュ>/` の下で、元のモデルは `original/`、
          置き換えたモデルは段取りが組んだ鍵の名前。`surrogate` は
          `runs/models/` のモデルを指す
- `bench` GPU の実測。`inputs` は細胞に順に割り当てる入力の並び (系列の掃引の点)、
          `model` は sim と同じで、細胞を `cells` の数ずつ
          並べて刻みごとに全細胞をそろえて GPU で回し、細胞の数ごとの `<細胞の数>/` に
          使ったツールの出力を加工せずにツールごとのファイルで書く
          (neurosurrogate.bench)。
          置き場は `runs/bench/<ハッシュ>/`

どれも `device` (cpu か gpu) を持つ。計算の順序が変わって結果がずれるので同一性に
入れ、JAX の backend が合わなければ回さない (backend は呼ぶ側が JAX_PLATFORMS で選ぶ)。
gpu でも GPU を使うのは AE の学習だけで、シミュレーションは CPU で回す。

書くもの (図が読む形。`.f64` は little-endian の float64 を並べただけ、
JSON の nan は null):

    spec.json  meta.json (kind = part / model / original / replaced / bench,
               name, created, elapsed = 計算にかかった秒, units = sim の電流パラメータの単位,
               env = 回した環境:
               backend、device、jax と python の版、host、コードとイメージの版)
    part       learner.joblib (圧縮器と右辺)、compressor/decode.json (潜在 →
               ゲートの曲線を潜在の軸ごとに。区画の型の状態の並びも)、
               compressor/plane.json (AE だけ。群のゲート 2 本ずつの平面の上で
               encoder が返す潜在の格子)、rhs/field.json (学習器の右辺と、元のモデルの
               潜在の速さの (V, z) の格子)、compressor/own.json と
               rhs/own.json (圧縮器と右辺に固有の中身。kind で形が変わる。PCA は
               向きと平均も)、fit/<comp_id>/ に {latent,recon_rmse}.f64 と
               summary.json (再構成の MSE と分散比)
    model      summary.json (組んだ 1 刻みの演算コスト。学習器の中身は学習器の
               run)。KAM のモデルは加えて kam.json (区画、吸収したゲート、群ごとの
               ゲートと潜在の名前と Kepler 1992 の式(6)の重みと式(10)の条件)、
               equivalent/ (学習の入力の区画の元のモデルの軌道を等価電位にしたもの:
               V、吸収したゲートと群のゲートの等価電位、群ごとの ψ。waves と同じ形)、
               reduced/ (同じ電流を縮約で回した waves)
    sim        waves/。置き換えたモデルは加えて、projected/<comp>/ (置換した comp の
               original/ のゲートを潜在へ射影したもの。waves と同じ形) と、その上の
               その上の潜在の傾きの正解 (射影の前進差分) slope/<comp>/ と、学習器の
               右辺が返す dz/dt の rhs/<comp>/ (どちらも同じ形)。
               run は生データだけを持ち、スパイクの特徴量 (spikes.py) と元のモデルとの差は API が
               頼まれるたびに波形から計算する
    bench      <細胞の数>/ に、bench.json (bench.py が自分で測った時間・状態の置き方・
               発火の回数)、jax.memory_stats.{placed,after}.json、nvidia-smi.{apps,gpu}.csv、
               ncu.csv (`ncu` が真なら)。どれもツールの出力のままで、集計は API が
               頼まれるたびに計算する (remote/read/bench.ts)

    waves/meta.json           t0, dt, 点ごとの長さ n (sim は入力 1 本なので点 0 だけ),
                              features [[comp_id, 変数, gate]],
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
import platform
import shutil
import sys
import time
from collections.abc import Hashable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
import xarray as xr

from .bench import OWN_FILE, cache, run_checked
from .physics.simulate import (
    EvalSeries,
    gate_matrix,
    i_internal,
    latent_dataset,
    potential,
    simulate,
)
from .store import assemble, load_model, part_config, save_part, train_root
from .surrogate.model import (
    Surrogate,
    applicable,
    describe_group,
    describe_kam,
    fit_group,
    group_latents,
    latent_slopes,
    replace,
    summary,
)

# spec の種類ごとの形式の版。段取りが spec に入れる = 形式を変えたらその種類の鍵が変わる
# (shared/spec.ts の FORMATS と揃える)
_FORMATS = {"part": 23, "model": 14, "sim": 14, "bench": 12}
_NOT_REPLACEABLE = 3
_RUNNING = 4
_CLAIM_GRACE = 60.0  # [s] 書きかけを作ってから pid を置くまでの猶予 (これより古く pid の無いものは残骸)
_DIVERGE_V = 1e3  # |V| [mV] の発散の閾値 (生理的な範囲は ±200 程度)
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


# --- 4 種類の run ---------------------------------------------------------------


def _names(series: EvalSeries) -> dict[str, int]:
    net = series.spec.net
    return {comp: net.index(comp) for comp in net.names}


def _part(root: Path, spec: dict) -> dict:
    """学習の入力の元のモデルの波形を読んで学習器 1 つを学習し、その中身を書く。"""
    runs = root.parents[1]
    config = part_config(runs, spec)
    [data] = _read_waves(train_root(runs, spec["input"]) / "waves")
    group = config.groups[0]
    part = fit_group(config, group, data)
    save_part(root, part)
    files, fits = describe_group(config, group, part, data)
    for path, value in files.items():
        _write_json(root / path, value)
    for comp_id, comp_fit in fits.items():
        folder = root / "fit" / str(comp_id)
        folder.mkdir(parents=True)
        _write_f64(folder / "latent.f64", comp_fit.latent.T)
        _write_f64(folder / "recon_rmse.f64", comp_fit.recon_rmse)
        _write_json(folder / "summary.json", comp_fit.summary)
    return {}


def _model(root: Path, spec: dict) -> dict:
    """学習器を組み、組んだモデルの演算コストを書く。KAM のモデルは加えて、学習の入力の
    元のモデルの軌道を等価電位にしたものと、同じ電流で回した縮約の軌道と、条件を書く。"""
    runs = root.parents[1]
    surrogate = assemble(runs, spec)
    _write_json(root / "summary.json", summary(surrogate))
    if "kam" in spec:
        [data] = _read_waves(train_root(runs, spec["kam"]["input"]) / "waves")
        files, equivalent, reduced = describe_kam(surrogate, data)
        names = {c.name: i for i, c in enumerate(surrogate.config._net().nodes)}
        _write_json(root / "kam.json", files)
        _write_waves(
            root / "equivalent", [equivalent], {files["comp"]: names[files["comp"]]}
        )
        _write_waves(root / "reduced", [reduced], names)
    return {}


def _project(surrogate: Surrogate, ds: xr.Dataset, comp_id: int) -> xr.Dataset:
    """元のモデルの区画のゲートを、各群の圧縮器で潜在へ落とした [V, 潜在]。"""
    config = surrogate.config
    latents, names = [], []
    for (compressor, _), group in zip(surrogate.parts, config.groups, strict=True):
        gates = gate_matrix(ds, comp_id)[:, config.positions(group)]
        latents.append(compressor.encode(gates))
        names.extend(group_latents(group))
    return latent_dataset(
        potential(ds, comp_id),
        np.concatenate(latents, axis=1),
        names,
        i_internal(ds, comp_id),
        comp_id,
        float(ds["time"][1] - ds["time"][0]),
    )


def _slopes(
    surrogate: Surrogate, ds: xr.Dataset, comp_id: int, net: Any
) -> tuple[xr.Dataset, xr.Dataset] | None:
    """元のモデルの区画の軌道の上の、潜在の傾きの正解と学習器の右辺 ([V, 潜在] の形)。"""
    slopes = latent_slopes(surrogate, ds, comp_id, net.nodes[comp_id].resolved_params)
    if slopes is None:
        return None
    names = [n for g in surrogate.config.groups for n in group_latents(g)]
    dt = float(ds["time"][1] - ds["time"][0])
    v, u = potential(ds, comp_id), i_internal(ds, comp_id)
    return (
        latent_dataset(v, slopes[0], names, u, comp_id, dt),
        latent_dataset(v, slopes[1], names, u, comp_id, dt),
    )


def _diverged(v: np.ndarray) -> bool:
    """電位が NaN・inf を含むか、生理的にあり得ない大きさへ振り切れたか。"""
    return not bool(np.all(np.isfinite(v))) or float(np.abs(v).max()) > _DIVERGE_V


def _sim(root: Path, spec: dict) -> dict:
    """入力をモデルで回して波形を書く。置き換えたモデルなら、
    置換した区画の射影も書く。"""
    model = spec["model"]
    targets = model["replace_targets"] if model else []
    series = EvalSeries.from_dict({**spec["input"], "replace_targets": targets})
    point = series.spec
    if model is None:
        _write_waves(root / "waves", [simulate(point.run_input())], _names(series))
        return {"units": series.units}
    runs = root.parents[2]
    surrogate = load_model(runs, model["surrogate"])
    if not applicable(surrogate.config, series):
        print(f"{model['surrogate']} は {targets} を置換できない")
        sys.exit(_NOT_REPLACEABLE)
    [original] = _read_waves(root.parent / "original/waves")
    ds = simulate(replace(surrogate, point.run_input(), targets))
    if broken := [c for c in targets if _diverged(potential(ds, point.net.index(c)))]:
        _logger.warning("置換系の電位が発散 (%s): %s", point.current_type, broken)
    _write_waves(root / "waves", [ds], _names(series))
    net = point.net
    for comp in targets:
        projected = _project(surrogate, original, net.index(comp))
        _write_waves(root / "projected" / comp, [projected])
        slopes = _slopes(surrogate, original, net.index(comp), net)
        if slopes is not None:
            _write_waves(root / "slope" / comp, [slopes[0]])
            _write_waves(root / "rhs" / comp, [slopes[1]])
    return {"units": series.units}


def _bench(root: Path, spec: dict) -> dict:
    """細胞の数ごとに別のプロセスで GPU の実測を回す。

    `<細胞の数>/` に、ツールごとの出力をそのまま書く。
    """
    runs = root.parents[1]
    gpu = {"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    env = {**os.environ, **gpu}
    command = [sys.executable, "-m", "neurosurrogate.bench", str(root / "spec.json")]
    for cells in spec["cells"]:
        out = root / str(cells)
        text = run_checked([*command, str(runs), str(cells), "--out", str(out)], env)
        row = json.loads(text.strip().splitlines()[-1])
        _write_json(out / OWN_FILE, row)
        if spec.get("ncu"):
            cache(root / "spec.json", runs, cells, row["compartments"], env, out)
        _logger.info("bench %s cells %s", root.name, cells)
    return {}


def _kind(spec: dict) -> str:
    """図に出す run の種類。sim はモデルの有無で、元のモデルと置き換えたモデルに
    分ける。"""
    if spec["kind"] != "sim":
        return str(spec["kind"])
    return "original" if spec["model"] is None else "replaced"


_KINDS = {"part": _part, "model": _model, "sim": _sim, "bench": _bench}


def _writer_alive(partial: Path) -> bool:
    """書きかけの run を書いているプロセスが生きているか (落ちたあとの残骸なら偽)。"""
    try:
        os.kill(int((partial / "pid").read_text()), 0)
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _claim(partial: Path, name: str) -> None:
    """書きかけの置き場を作って pid を置く。生きた書き手か、作られたばかりで pid をまだ置いていない書き手が
    いれば退く (同じ元のモデルを二つの評価が同時に回すとき、相手の書きかけを消さない)。残骸だけを片付けて取り直す。"""
    for _ in range(2):
        try:
            partial.mkdir(parents=True)
        except FileExistsError:
            try:
                fresh = (
                    not (partial / "pid").exists()
                    and time.time() - partial.stat().st_mtime < _CLAIM_GRACE
                )
            except FileNotFoundError:  # 相手が書き終えて名前を変えた
                continue
            if _writer_alive(partial) or fresh:
                print(f"{name} は別のプロセスが書いている")
                sys.exit(_RUNNING)
            shutil.rmtree(partial, ignore_errors=True)
            continue
        (partial / "pid").write_text(str(os.getpid()))
        return
    sys.exit(_RUNNING)


def _check_device(spec: dict) -> None:
    """spec の device (cpu か gpu) と、この JAX の backend が合うか。選ぶのは呼ぶ側 (JAX_PLATFORMS)。
    gpu の学習でも、1 細胞を順に進めるシミュレーションは CPU が速いので既定を CPU にし、GPU は AE の学習だけが使う。"""
    if jax.default_backend() != spec["device"]:
        raise ValueError(
            f"spec は {spec['device']} で回す run だが、"
            f"JAX の backend は {jax.default_backend()}"
        )
    if spec["device"] == "gpu":
        jax.config.update("jax_default_device", jax.devices("cpu")[0])


def _env() -> dict:
    """回した環境 (同一性に入らない)。コードとイメージの版は段取りが環境変数で渡す。"""
    return {
        "backend": jax.default_backend(),
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "python": platform.python_version(),
        "host": os.environ.get("SINDY_HOST", platform.node()),
        "code": os.environ.get("SINDY_CODE"),
        "image": os.environ.get("SINDY_IMAGE"),
    }


def _run(spec_path: Path, root: Path, name: str) -> None:
    """spec を回して `root` を埋める。`root` があれば回さない。"""
    if root.exists():
        _logger.info("%s は回してある", root.name)
        return
    spec = json.loads(spec_path.read_text())
    want = _FORMATS[spec["kind"]]
    if spec.get("format") != want:
        raise ValueError(
            f"{spec['kind']} の spec の形式 {spec.get('format')} は {want} でない"
        )
    _check_device(spec)
    partial = root.with_name(f"{root.name}.partial")
    _claim(partial, root.name)
    try:
        (partial / "spec.json").write_text(spec_path.read_text())
        start = time.monotonic()
        extra = _KINDS[spec["kind"]](partial, spec)
        meta = {
            "kind": _kind(spec),
            "name": name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "elapsed": round(time.monotonic() - start, 1),
            "env": _env(),
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
