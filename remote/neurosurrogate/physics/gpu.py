"""細胞を並べて、刻みごとに全細胞をそろえて GPU で解く (Pallas の Triton)。

1 スレッドが 1 区画を持つ。区画ごとに値 (密度・寸法・型) が違うので、区画ごとに
カーネルを分け、1 刻みでは区画の数だけカーネルを起こす (全細胞のその区画を並べて回す)。
全区画がその刻みを終えてから次の刻みへ進む。

刻みをまたいで持ち越す状態 (各区画の V、潜在 z、縮約しなかったゲート) はすべて VRAM
(global memory) に変数ごとの配列 (SoA) で置き、毎刻み読み込んで書き戻す。隣の区画の
V は前の刻みの値を VRAM から読む (前進 Euler。CPU の simulate と同じ)。刻みの中だけで
使う値 (z から作り直すゲート、SINDy の項、レート) はカーネルの中の局所の値 (レジスタ)。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from .cells import ColumnCell

# 電流の表の列 (系列の点) の数の上限。Triton のブロックは 2 の冪にそろえる。
POINTS = 8
# 1 つのプログラム (スレッドの束) が持つ細胞の数。
BLOCK = 128
# 1 刻みのカーネルの名前の頭 (区画の番号を後ろに付ける。ncu はこの頭で拾う)。
KERNEL = "step_comp"


def _floats(params: Any) -> Any:
    """params の値を Python の数にする (カーネルが配列の定数を抱え込まないように)。"""
    if params is None or not hasattr(params, "_fields"):
        return params
    return params._replace(**{f: float(getattr(params, f)) for f in params._fields})


def currents_table(points: list[np.ndarray]) -> np.ndarray:
    """系列の点ごとの電流を (時刻 × POINTS) の表にする。空いた列は 0。"""
    if len(points) > POINTS:
        raise ValueError(f"系列の点は {POINTS} 個まで")
    length = min(len(c) for c in points)
    table = np.zeros((length, POINTS))
    for k, c in enumerate(points):
        table[:, k] = c[:length]
    return table


@dataclass(frozen=True)
class Layout:
    """VRAM に置く持ち越す状態の並びと、1 刻みの読み書きの量の予想 [バイト]。"""

    names: list[tuple[int, str]]  # (区画の番号, 変数の名前)
    state_bytes: int
    step_load_bytes: int
    step_store_bytes: int


@dataclass(frozen=True)
class _Comp:
    """1 区画のカーネルを組むもの。neighbors は流入元の区画で、自分を必ず含む。"""

    index: int
    cell: ColumnCell
    neighbors: list[int]
    weights: list[float]
    area: float
    stimulated: bool
    recorded: bool


def _comp_call(c: _Comp, dt: float, n_points: int, cells: int, threshold: float):
    """1 区画の 1 刻みのカーネル。流入電流は Σ_j L[j,i] V_j / 面積 + 注入。"""
    p, kernel, states = _floats(c.cell.params), c.cell.kernel, c.cell.states
    nb, ns = len(c.neighbors), len(states)
    own = c.neighbors.index(c.index)

    def body(u_ref, rows_ref, *refs):
        # 入力: 流入元の V、自分の状態、(記録する区画なら) 発火の回数
        # 出力: 自分の新しい V、自分の状態 (入力と同じ置き場)、発火の回数
        v_in, refs = refs[:nb], refs[nb:]
        s_in, refs = refs[:ns], refs[ns:]
        count_in, refs = (refs[0], refs[1:]) if c.recorded else (None, refs)
        v_out, refs = refs[0], refs[1:]
        s_out, refs = refs[:ns], refs[ns:]
        row = rows_ref[...]
        coupled = jnp.zeros(row.shape, jnp.float64)
        for ref, weight in zip(v_in, c.weights, strict=True):
            coupled = coupled + ref[...] * weight
        u = coupled / c.area
        if c.stimulated:
            for k in range(n_points):
                u = jnp.where(row == k, u + u_ref[k], u)
        v = v_in[own][...]
        s = {n: ref[...] for n, ref in zip(states, s_in, strict=True)}
        dv, ds = kernel(p, u, v, s)
        nxt = v + dv * dt
        v_out[...] = nxt
        for n, ref in zip(states, s_out, strict=True):
            ref[...] = s[n] + ds[n] * dt
        if c.recorded:
            line = jnp.full(v.shape, threshold, v.dtype)
            crossed = ((v < line) & (nxt >= line)).astype(jnp.int32)
            refs[0][...] = count_in[...] + crossed

    block = pl.BlockSpec((BLOCK,), lambda b: (b,))
    one = jax.ShapeDtypeStruct((cells,), jnp.float64)
    count = [jax.ShapeDtypeStruct((cells,), jnp.int32)] if c.recorded else []
    n_in = 2 + nb + ns + len(count)
    # 自分の状態と発火の回数は、出力が入力と同じ置き場を書き換える (V は前の刻みを残す)
    aliases = {2 + nb + k: 1 + k for k in range(ns)}
    if c.recorded:
        aliases[2 + nb + ns] = 1 + ns
    return pl.pallas_call(
        body,
        out_shape=(one, *[one] * ns, *count),
        grid=(cells // BLOCK,),
        in_specs=[pl.BlockSpec((POINTS,), lambda b: (0,)), *[block] * (n_in - 1)],
        out_specs=tuple([block] * (1 + ns + len(count))),
        input_output_aliases=aliases,
        compiler_params=plt.CompilerParams(num_warps=4),
        name=f"{KERNEL}{c.index:02d}",
    )


def stepped_net(
    cells_of: list[ColumnCell],
    laplacian: np.ndarray,
    areas: np.ndarray,
    stim: int,
    record: int,
    dt: float,
    n_points: int,
    cells: int,
    threshold: float = 0.0,
) -> tuple[Callable, Layout]:
    """区画のネットを、細胞を並べて刻みごとに解く jit 済みの関数 fn(電流の表, 行)。

    cells_of は区画ごとの型、stim は電流を入れる区画、record は発火を数える区画。
    返り値は (最後の区画ごとの V, 区画ごとの状態の並び, 発火の回数) と並び方。細胞 k は
    行 rows[k] の電流で回る。cells は BLOCK の倍数。
    """
    if cells % BLOCK != 0:
        raise ValueError(f"細胞の数は {BLOCK} の倍数")
    comps = []
    for i, cell in enumerate(cells_of):
        neighbors = [j for j in range(len(cells_of)) if laplacian[j, i] != 0 or j == i]
        weights = [float(laplacian[j, i]) for j in neighbors]
        comps.append(
            _Comp(i, cell, neighbors, weights, float(areas[i]), i == stim, i == record)
        )
    calls = [_comp_call(c, dt, n_points, cells, threshold) for c in comps]

    def fn(table, rows):
        vs = [jnp.full((cells,), float(c.cell.inits[0]), jnp.float64) for c in comps]
        states = []
        for c in comps:
            inits = c.cell.inits[1:]
            states.append([jnp.full((cells,), float(x), jnp.float64) for x in inits])
        count = jnp.zeros((cells,), jnp.int32)

        def step(t, carry):
            vs, states, count = carry
            new_vs, new_states = [], []
            for c, call in zip(comps, calls, strict=True):
                inputs = [vs[j] for j in c.neighbors] + states[c.index]
                out = call(table[t], rows, *inputs, *([count] if c.recorded else []))
                ns = len(c.cell.states)
                new_vs.append(out[0])
                new_states.append(list(out[1 : 1 + ns]))
                if c.recorded:
                    count = out[1 + ns]
            return new_vs, new_states, count

        last = table.shape[0] - 1
        return jax.lax.fori_loop(0, last, step, (vs, states, count))

    return jax.jit(fn), _layout(comps, cells)


def _layout(comps: list[_Comp], cells: int) -> Layout:
    """持ち越す状態の並びと、1 刻みで VRAM から読み、書き戻す量の予想。"""
    names: list[tuple[int, str]] = []
    load = store = 0
    for c in comps:
        names.append((c.index, "V"))
        names.extend((c.index, n) for n in c.cell.states)
        ns = len(c.cell.states)
        # 読む: 流入元の V、自分の状態、細胞の行 (int32)、電流の表の 1 行、発火の回数
        load += cells * (8 * len(c.neighbors) + 8 * ns + 4)
        load += (cells // BLOCK) * POINTS * 8 if c.stimulated else 0
        # 書く: 自分の V と状態、発火の回数
        store += cells * (8 + 8 * ns)
        if c.recorded:
            load += cells * 4
            store += cells * 4
    return Layout(names, cells * 8 * len(names), load, store)
