"""細胞を並べて、刻みごとに全細胞をそろえて GPU で解く (Pallas の Triton)。

1 刻みがカーネル 1 回で、全細胞がその刻みを終えてから次の刻みへ進む (ネットワークで
V をやり取りする区切りがここにできる)。刻みをまたいで持ち越す状態 (V、潜在 z、縮約
しなかったゲート) はすべて VRAM (global memory) に置き、毎刻み読み込んで書き戻す。
刻みの中だけで使う値 (z から作り直すゲート、SINDy の項、レート) はカーネルの中の
局所の値 (レジスタ) に置く。区画は単一区画の ColumnCell (配列にまとめない右辺) だけ。
"""

from collections.abc import Callable
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
# 1 刻みのカーネルの名前 (ncu がこの名前で拾う)。
KERNEL = "step_cells"


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


def stepped_batch(
    cell: ColumnCell, dt: float, n_points: int, cells: int, threshold: float = 0.0
) -> tuple[Callable, list[str]]:
    """刻みごとに全細胞をそろえて解く jit 済みの関数 fn(電流の表, 細胞ごとの行)。

    返り値は (最後の持ち越す状態の並び [V, *states], 発火の回数) と、持ち越す状態の
    名前の並び。細胞 k は行 rows[k] の電流で回る。cells は BLOCK の倍数。
    """
    if cells % BLOCK != 0:
        raise ValueError(f"細胞の数は {BLOCK} の倍数")
    p, kernel, states = _floats(cell.params), cell.kernel, cell.states
    names = ["V", *states]

    k = len(names)

    def body(u_ref, rows_ref, count_in, *refs):
        # 入力の持ち越す状態 (VRAM から読む)、出力 (同じ置き場へ書き戻す)、発火の回数
        old, new, count_out = refs[:k], refs[k : 2 * k], refs[2 * k]
        row = rows_ref[...]
        u = jnp.zeros(row.shape, jnp.float64)
        for j in range(n_points):
            u = jnp.where(row == j, u_ref[j], u)
        v = old[0][...]
        s = {n: ref[...] for n, ref in zip(states, old[1:], strict=True)}
        dv, ds = kernel(p, u, v, s)
        nxt = v + dv * dt
        line = jnp.full(v.shape, threshold, v.dtype)
        crossed = ((v < line) & (nxt >= line)).astype(jnp.int32)
        new[0][...] = nxt
        for n, ref in zip(states, new[1:], strict=True):
            ref[...] = s[n] + ds[n] * dt
        count_out[...] = count_in[...] + crossed

    block = pl.BlockSpec((BLOCK,), lambda b: (b,))
    state = jax.ShapeDtypeStruct((cells,), jnp.float64)
    # 入力は (電流, 行, 発火の回数, *状態)。状態と回数は出力と同じ置き場を書き換える
    aliases = {3 + i: i for i in range(k)} | {2: k}
    step_call = pl.pallas_call(
        body,
        out_shape=(*[state] * k, jax.ShapeDtypeStruct((cells,), jnp.int32)),
        grid=(cells // BLOCK,),
        in_specs=[pl.BlockSpec((POINTS,), lambda b: (0,)), block, block, *[block] * k],
        out_specs=tuple([block] * (k + 1)),
        input_output_aliases=aliases,
        compiler_params=plt.CompilerParams(num_warps=4),
        name=KERNEL,
    )

    def fn(table, rows):
        init = [jnp.full((cells,), float(x), jnp.float64) for x in cell.inits]
        count = jnp.zeros((cells,), jnp.int32)

        def step(t, carry):
            values, count = carry
            out = step_call(table[t], rows, count, *values)
            return list(out[:k]), out[k]

        values, count = jax.lax.fori_loop(0, table.shape[0] - 1, step, (init, count))
        return values, count

    return jax.jit(fn), names
