"""GPU の実測: 細胞を並べ、刻みごとに全細胞をそろえて回し、置き場ごとの量を測る。

`uv run python -m neurosurrogate.bench <spec.json> <runs> <細胞の数> [--once]`。
spec は run.py の `bench` の spec (入力と、元のモデルか学習 run)。1 回の呼び出しで
1 つの細胞の数だけを測り、結果を JSON で標準出力に書く。細胞の数ごとに別のプロセスで
呼ぶのは、XLA の先取りを切って VRAM を細胞の数ごとに測るためと、Nsight Compute (ncu)
をかぶせるため。`--once` は ncu の下で 1 回だけ回す (時間は測らない)。

1 刻みがカーネル 1 回 (physics.gpu)。刻みをまたいで持ち越す状態 (V、潜在 z、縮約
しなかったゲート) は VRAM に置いて毎刻み読み書きし、刻みの中だけで使う値はレジスタに
置く。ncu では途中の 1 刻みのカーネルを測り、VRAM からの読み込みと書き戻しの量を、
持ち越す状態の数から出した予想と並べる。レジスタの数と溢れ (local memory) も測る。
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
from .physics.gpu import BLOCK, KERNEL, POINTS, currents_table, stepped_batch
from .physics.simulate import EvalSeries, RunInput
from .surrogate.model import Surrogate, replace, surrogate_column_cell

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
# ncu の指標の名前 → bench.json の鍵。
NCU_KEYS = {
    "launch__registers_per_thread": "registers_per_thread",
    "launch__thread_count": "threads",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum": "global_store_bytes",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum": "global_load_bytes",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum": "local_store_bytes",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum": "local_load_bytes",
    "dram__bytes_write.sum": "dram_write_bytes",
    "dram__bytes_read.sum": "dram_read_bytes",
    "l1tex__t_sector_hit_rate.pct": "l1_hit_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_pct",
}


def batch_input(
    spec: dict, runs: Path
) -> tuple[RunInput, ColumnCell, list[np.ndarray]]:
    """spec の入力を、ネットと GPU 用の区画と、系列の点ごとの電流にする。"""
    model = spec["model"]
    targets = model["replace_targets"] if model else []
    series = EvalSeries.from_dict({**spec["input"], "replace_targets": targets})
    inputs = [point.run_input() for point in series.points]
    if len(inputs[0].net.nodes) != 1:
        raise ValueError("GPU の実測は単一区画だけ")
    if model is None:
        cell = column_cell(inputs[0].net.nodes[0])
    else:
        surrogate = Surrogate.load(runs / "models" / model["train"] / "surrogate")
        inputs = [replace(surrogate, run, targets) for run in inputs]
        cell = surrogate_column_cell(surrogate, inputs[0].net.nodes[0])
    return inputs[0], cell, [run.current for run in inputs]


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


def _smi_used() -> float | None:
    """nvidia-smi が示すこのプロセスの GPU のメモリ [MiB]。読めなければ None。"""
    query = ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(
            ["nvidia-smi", *query], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in out.strip().splitlines():
        pid, used = (x.strip() for x in line.split(","))
        if int(pid) == os.getpid():
            return float(used)
    return None


def _vram_total() -> int | None:
    """GPU の VRAM の全容量 [バイト] (nvidia-smi)。読めなければ None。"""
    query = ["--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(
            ["nvidia-smi", *query], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return int(float(out.strip().splitlines()[0]) * 1024 * 1024)


def measure(spec: dict, runs: Path, cells: int, once: bool = False) -> dict:
    """細胞を並べて刻みごとに回し、時間・置き場ごとの量・発火の回数を返す。"""
    gpu = jax.devices("gpu")[0]
    run, cell, points = batch_input(spec, runs)
    fn, names = stepped_batch(cell, run.dt, len(points), cells)
    # プロファイルに要るのは 1 刻みのカーネルだけなので、系列の頭の STEPS 刻みで止める
    table = jax.device_put(jnp.array(currents_table(points)[: STEPS + 1]), gpu)
    rows = jax.device_put(jnp.arange(cells, dtype=jnp.int32) % len(points), gpu)
    jax.block_until_ready((table, rows))
    placed = (gpu.memory_stats() or {}).get("bytes_in_use")
    if once:
        # ncu は測らない起動にも割り込むので、測る刻み (101 回目) の少し先までで止める
        jax.block_until_ready(fn(table, rows))
        return {"cells": cells}
    start = time.perf_counter()
    jax.block_until_ready(fn(table, rows))  # コンパイルを含む 1 回目
    first = time.perf_counter() - start
    start = time.perf_counter()
    _, counts = jax.block_until_ready(fn(table, rows))
    wall = time.perf_counter() - start
    stats = gpu.memory_stats() or {}
    peak = stats.get("peak_bytes_in_use")
    steps = table.shape[0] - 1
    counts = np.asarray(counts)
    n = len(points)
    per_point = [sorted({int(c) for c in counts[k::n]}) for k in range(min(n, cells))]
    # 1 刻みで読み書きする量: 細胞ごとに持ち越す状態と発火の回数 (読み込みは行も)、
    # プログラムごとに電流の表の 1 行 (POINTS 個の float64)
    state = len(names) * 8
    table_row = cells // BLOCK * POINTS * 8
    return {
        "cells": cells,
        "steps": steps,
        "compile_s": first - wall,
        "wall_s": wall,
        "cell_steps_per_s": cells * steps / wall,
        "vram_states": names,
        "latent": cell.latent,
        "decoded": cell.decoded,
        "vram_state_bytes": cells * state,
        "expected_step_load_bytes": cells * (state + 4 + 4) + table_row,
        "expected_step_store_bytes": cells * (state + 4),
        "peak_bytes": peak,
        "input_bytes": placed,
        "work_bytes": None if peak is None or placed is None else peak - placed,
        "smi_used_mib": _smi_used(),
        "vram_total_bytes": _vram_total(),
        "spike_counts": per_point,
    }


def _rows(out: str) -> dict[str, dict[str, float]]:
    """ncu の CSV を、カーネルの ID → 指標の名前 → 値 にする。"""
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


def cache(spec_path: Path, runs: Path, cells: int, env: dict) -> dict:
    """ncu の下で回し、途中の 1 刻みのカーネルの置き場ごとの量を測る。"""
    command = [
        *(_ncu(), "--csv", "--print-units", "base"),
        *("--kernel-name", f"regex:{KERNEL}"),
        *("--metrics", ",".join(NCU_METRICS), "--target-processes", "application-only"),
        *("--launch-skip", str(SKIP), "--launch-count", "1"),
        *(sys.executable, "-m", "neurosurrogate.bench"),
        *(str(spec_path), str(runs), str(cells), "--once"),
    ]
    for attempt in range(NCU_ATTEMPTS):
        try:
            rows = _rows(run_checked(command, env, timeout=NCU_TIMEOUT))
            break
        except RuntimeError:
            # ncu はときどきセグメンテーション違反で落ちるので、やり直す
            if attempt == NCU_ATTEMPTS - 1:
                raise
    if len(rows) != 1:
        raise RuntimeError(
            f"ncu が 1 刻みのカーネルを {len(rows)} 本拾った (1 本のはず)"
        )
    [row] = rows.values()
    return {key: row.get(metric) for metric, key in NCU_KEYS.items()}


def main(argv: list[str]) -> None:
    """細胞の数 1 つを測って JSON で書く。"""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    spec_path, runs, cells = Path(argv[0]), Path(argv[1]), int(argv[2])
    spec = json.loads(spec_path.read_text())
    print(json.dumps(measure(spec, runs, cells, once="--once" in argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
