"""GPU の実測: 細胞を並べ、刻みごとに全細胞をそろえて回し、置き場ごとの量を測る。

`uv run python -m neurosurrogate.bench <spec.json> <runs> <細胞の数>
[--out <dir>] [--once]`。
spec は run.py の `bench` の spec (入力と、元のモデルか学習器を組んだモデル)。1 回の呼び出しで
1 つの細胞の数だけを測り、自分で測ったもの (時間、状態の置き方、発火の回数) を JSON で
標準出力に書く。`--out` を渡せば、使ったツールの出力を加工せずにそのディレクトリへ
ツールごとのファイルで書く (JAX の memory_stats、nvidia-smi の CSV)。ncu の CSV は
ncu をかぶせる側 (`cache`) が同じディレクトリに書く。集計は API (remote/read/bench.ts)
が頼まれるたびにこれらから計算する。細胞の数ごとに別のプロセスで呼ぶのは、XLA の先取りを
切って VRAM を細胞の数ごとに測るためと、Nsight Compute (ncu) をかぶせるため。`--once` は
ncu の下で 1 回だけ回す (時間は測らない)。

1 刻みがカーネル 1 回 (physics.gpu)。刻みをまたいで持ち越す状態 (V、潜在 z、縮約
しなかったゲート) は VRAM に置いて毎刻み読み書きし、刻みの中だけで使う値はレジスタに
置く。ncu では途中の 1 刻みのカーネルを測る (VRAM からの読み込みと書き戻しの量、
レジスタの数と溢れ (local memory))。持ち越す状態の数から出した量の予想は自分で書く。
細胞 i には系列の点 i mod 点の数 の電流を入れる。発火の回数は、同じ電流の細胞どうしで
揃うかの確かめに使う。
"""

import csv
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .physics.cells import ColumnCell, column_cell
from .physics.gpu import KERNEL, currents_table, stepped_net
from .physics.simulate import EvalSeries, RunInput, coupling
from .store import load_model
from .surrogate.model import replace, surrogate_column_cell

# ncu を 1 回待つ秒数と、落ちたときにやり直す回数。
NCU_TIMEOUT = 120
NCU_ATTEMPTS = 3
# 回す刻みの数。ncu はこのうち SKIP 回目の先の 1 回だけを測る。
STEPS = 128
SKIP = 100
# ncu で測る指標: レジスタ、溢れ (local)、global の書き込みと読み込み、DRAM、当たり率。
NCU_METRICS = [
    "launch__registers_per_thread",
    "launch__thread_count",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "dram__bytes_write.sum",
    "dram__bytes_read.sum",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
]
# ツールごとの出力のファイルの名前 (細胞の数のディレクトリの下)。
# shared/runs.ts の BENCH と揃える。
OWN_FILE = "bench.json"
NCU_FILE = "ncu.csv"
SMI_APPS_FILE = "nvidia-smi.apps.csv"
SMI_GPU_FILE = "nvidia-smi.gpu.csv"
JAX_PLACED_FILE = "jax.memory_stats.placed.json"
JAX_AFTER_FILE = "jax.memory_stats.after.json"


def batch_input(
    spec: dict, runs: Path
) -> tuple[RunInput, list[ColumnCell], list[np.ndarray]]:
    """spec の入力を、ネットと区画ごとの GPU 用の型と、系列の点ごとの電流にする。"""
    model = spec["model"]
    targets = model["replace_targets"] if model else []
    inputs = [
        EvalSeries.from_dict({**d, "replace_targets": targets}).spec.run_input()
        for d in spec["inputs"]
    ]
    surrogate = None
    if model is not None:
        surrogate = load_model(runs, model["surrogate"])
        inputs = [replace(surrogate, run, targets) for run in inputs]
    cells = []
    for comp in inputs[0].net.nodes:
        if surrogate is not None and comp.name in targets:
            cells.append(surrogate_column_cell(surrogate, comp))
        else:
            cells.append(column_cell(comp))
    return inputs[0], cells, [run.current for run in inputs]


def run_checked(
    command: list[str], env: dict | None = None, timeout: float | None = None
) -> str:
    """子プロセスを回して標準出力を返す。落ちたら標準エラーを添えて例外にする。

    timeout を過ぎたら、子が起こした孫ごと止める (ncu が落ちると孫が待ち続けるため)。
    """
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    ) as child:
        try:
            out, err = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
            message = f"{command[0]} が {timeout} 秒で終わらなかった"
            raise RuntimeError(message) from None
    if child.returncode != 0:
        if _alive_group(child.pid):
            os.killpg(child.pid, signal.SIGKILL)
        message = f"{command[0]} が {child.returncode} で落ちた:\n{err[-4000:]}"
        raise RuntimeError(message)
    return out


def _alive_group(pgid: int) -> bool:
    """プロセスの組がまだ残っているか。"""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _smi(query: str) -> str | None:
    """nvidia-smi の問い合わせの CSV (見出しと単位つき) をそのまま返す。

    読めなければ None。
    """
    try:
        return subprocess.run(
            ["nvidia-smi", query, "--format=csv"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _write_raw(out: Path, name: str, text: str | None) -> None:
    """ツールの出力 1 つを、加工せずにファイルへ書く。出力が無ければ書かない。"""
    if text is not None:
        (out / name).write_text(text)


def measure(
    spec: dict, runs: Path, cells: int, out: Path | None = None, once: bool = False
) -> dict:
    """細胞を並べて刻みごとに回し、自分で測ったものを返す。

    自分で測ったものは、時間・状態の置き方・発火の回数。

    out を渡せば、JAX の memory_stats (入力を置いた後と回した後) と nvidia-smi の CSV
    (このプロセスの使用量と GPU の全容量) を、加工せずにそこへ書く。
    """
    gpu = jax.devices("gpu")[0]
    run, types, points = batch_input(spec, runs)
    net = run.net
    laplacian, areas = coupling(net)
    record = net.index(net.names[run.stim_idx])
    fn, layout = stepped_net(
        types, laplacian, areas, run.stim_idx, record, run.dt, len(points), cells
    )
    # プロファイルに要るのは 1 刻みのカーネルだけなので、系列の頭の STEPS 刻みで止める
    table = jax.device_put(jnp.array(currents_table(points)[: STEPS + 1]), gpu)
    rows = jax.device_put(jnp.arange(cells, dtype=jnp.int32) % len(points), gpu)
    jax.block_until_ready((table, rows))
    placed = gpu.memory_stats()
    if once:
        jax.block_until_ready(fn(table, rows))
        return {"cells": cells}
    start = time.perf_counter()
    jax.block_until_ready(fn(table, rows))  # コンパイルを含む 1 回目
    first = time.perf_counter() - start
    start = time.perf_counter()
    _, _, counts = jax.block_until_ready(fn(table, rows))
    wall = time.perf_counter() - start
    after = gpu.memory_stats()
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        _write_raw(out, JAX_PLACED_FILE, None if placed is None else json.dumps(placed))
        _write_raw(out, JAX_AFTER_FILE, None if after is None else json.dumps(after))
        _write_raw(out, SMI_APPS_FILE, _smi("--query-compute-apps=pid,used_memory"))
        _write_raw(out, SMI_GPU_FILE, _smi("--query-gpu=memory.total"))
    steps = table.shape[0] - 1
    counts = np.asarray(counts)
    n = len(points)
    per_point = [sorted({int(c) for c in counts[k::n]}) for k in range(min(n, cells))]
    comp_states = {}
    decoded = {}
    for comp, cell in zip(net.nodes, types, strict=True):
        comp_states[comp.name] = ["V", *cell.states]
        if cell.decoded:
            decoded[comp.name] = cell.decoded
    return {
        "cells": cells,
        "compartments": len(net.nodes),
        "steps": steps,
        "compile_s": first - wall,
        "wall_s": wall,
        "comp_states": comp_states,
        "decoded": decoded,
        "vram_state_bytes": layout.state_bytes,
        "expected_step_load_bytes": layout.step_load_bytes,
        "expected_step_store_bytes": layout.step_store_bytes,
        "spike_counts": per_point,
        "pid": os.getpid(),
    }


def _rows(out: str) -> dict[str, dict[str, float]]:
    """ncu の CSV を、カーネルの ID → 指標の名前 → 値 にする。

    拾ったカーネルの数を確かめるためだけに読む。
    """
    lines = [line for line in out.splitlines() if line.startswith('"')]
    records = list(csv.reader(lines))
    if not records:
        return {}
    header = records[0]
    rows: dict[str, dict[str, float]] = {}
    for record in records[1:]:
        row = dict(zip(header, record, strict=False))
        try:
            value = float(row["Metric Value"].replace(",", ""))
        except (KeyError, ValueError):  # n/a や欠けた行
            continue
        rows.setdefault(row["ID"], {})[row["Metric Name"]] = value
    return rows


def _ncu() -> str:
    """ncu のパス。PATH に無ければ CUDA のイメージの置き場 (/opt/nvidia) から探す。"""
    found = shutil.which("ncu") or next(
        iter(sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"), reverse=True)), None
    )
    if found is None:
        raise FileNotFoundError("ncu (Nsight Compute) が見つからない")
    return found


def cache(
    spec_path: Path, runs: Path, cells: int, kernels: int, env: dict, out: Path
) -> None:
    """ncu の下で回し、途中の 1 刻みのカーネルの指標の CSV を out に書く。

    区画ごとのカーネルの行を、加工せずにそのまま書く。
    """
    command = [
        *(_ncu(), "--csv", "--print-units", "base"),
        *("--kernel-name", f"regex:{KERNEL}"),
        *("--metrics", ",".join(NCU_METRICS), "--target-processes", "application-only"),
        *("--launch-skip", str(SKIP * kernels), "--launch-count", str(kernels)),
        *(sys.executable, "-m", "neurosurrogate.bench"),
        *(str(spec_path), str(runs), str(cells), "--once"),
    ]
    for attempt in range(NCU_ATTEMPTS):
        try:
            text = run_checked(command, env, timeout=NCU_TIMEOUT)
            rows = _rows(text)
            break
        except RuntimeError:
            # ncu はときどきセグメンテーション違反で落ちるので、やり直す
            if attempt == NCU_ATTEMPTS - 1:
                raise
    if len(rows) != kernels:
        found = f"{len(rows)} 本 ({kernels} 本のはず)"
        raise RuntimeError(f"ncu が 1 刻みのカーネルを {found} 拾った")
    _write_raw(out, NCU_FILE, text)


def main(argv: list[str]) -> None:
    """細胞の数 1 つを測って JSON で書く。"""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    spec_path, runs, cells = Path(argv[0]), Path(argv[1]), int(argv[2])
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    spec = json.loads(spec_path.read_text())
    print(json.dumps(measure(spec, runs, cells, out=out, once="--once" in argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
