"""**1 ペアの詳細図**: 入力電流プレビューと、原系/置換系の比較 (波形・差分・相平面)。

どのペアを描くかは呼び出し側が選び、ここは Dataset だけを受ける (結果型
`SimResult`/`SeriesResults` を知らない)。一括生成する `cell_figs` は
`waveform/__init__.py`。marimo 非依存。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from ..core import access
from ..core.access import POTENTIAL_VAR
from ..plotting import PanelSpec, TraceSpec, error_fig, new_figure, place_legend

if TYPE_CHECKING:
    from ..sim.spec import SimSpec


def current_preview_fig(spec: SimSpec) -> Figure:
    """電流波形プレビュー。構築失敗は error_fig。marimo 非依存。"""
    try:
        i_ext = spec.current()
    except Exception as e:  # noqa: BLE001
        return error_fig(f"build failed: {e}")
    t = np.arange(len(i_ext)) * spec.dt
    fig = new_figure(figsize=(6, 2))
    ax = fig.subplots()
    ax.plot(t, i_ext, lw=0.8)
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("I_ext [μA/cm²]")
    ax.set_title(f"{spec.current_type} preview")
    return fig


def panels_simple(
    ds: xr.Dataset, comps: Sequence[int] | None = None
) -> list[PanelSpec]:
    """全 comp の波形。comps を渡すとその comp だけに絞る (None=全部)。
    traub19 のような多 comp モデルは全部重ねると読めないため。"""
    comp_ids = [int(i) for i in access.comp_ids(ds) if comps is None or int(i) in comps]
    multi = len(comp_ids) > 1
    spec: list[PanelSpec] = [
        PanelSpec("I_ext", [TraceSpec(*access.i_ext(ds))]),
    ]

    if access.has_i_internal(ds):
        spec.append(
            PanelSpec(
                "I_internal",
                [
                    TraceSpec(*access.i_internal(ds, i), label=f"Comp {i}")
                    for i in comp_ids
                ],
            )
        )

    spec.append(
        PanelSpec(
            "v(t) [mV]",
            [
                TraceSpec(
                    *access.trace(ds, i, POTENTIAL_VAR),
                    label=f"v (Comp {i})" if multi else None,
                )
                for i in comp_ids
            ],
        )
    )

    gate_traces = [
        TraceSpec(*access.trace(ds, i, v), label=f"{v} (Comp {i})")
        for i in comp_ids
        for v in access.gate_variables(ds, i)
    ]
    if gate_traces:
        spec.append(PanelSpec("Gates / Latent", gate_traces))

    return spec


def panels_diff(
    original: xr.Dataset,
    preprocessed: xr.Dataset,
    surrogate: xr.Dataset,
    comp_id: int,
    i_ext_ylim: tuple[float, float] | None = None,
) -> list[PanelSpec]:
    return [
        PanelSpec(
            "I_ext(t)\n[μA/cm²]",
            [TraceSpec(*access.i_ext(original), color="gold")],
            # 未指定は評価刺激が読める発表用レンジ (train_raw と共有すると学習パルスの
            # 最大値に引っ張られ、評価の step が潰れる)。
            ylim=i_ext_ylim or (0.0, 5.0),
        ),
        PanelSpec(
            "v(t) [mV]",
            [
                TraceSpec(
                    *access.trace(original, comp_id, POTENTIAL_VAR),
                    label="orig v",
                    color="blue",
                ),
                TraceSpec(
                    *access.trace(surrogate, comp_id, POTENTIAL_VAR),
                    label="surr v",
                    color="red",
                    style="--",
                ),
            ],
        ),
        *[
            PanelSpec(
                latent,
                [
                    TraceSpec(
                        *access.trace(preprocessed, comp_id, latent),
                        label=f"orig {latent}",
                        color="blue",
                    ),
                    TraceSpec(
                        *access.trace(surrogate, comp_id, latent),
                        label=f"surr {latent}",
                        color="red",
                        style="--",
                    ),
                ],
            )
            for latent in access.latent_variables(preprocessed)
        ],
    ]


def attractor_fig(orig_ds: xr.Dataset, surr_ds: xr.Dataset, comp_id: int) -> Figure:
    """相平面 (V × 第1潜在) の重ね描き。原系と置換系のダイナミクス一致度を見る。
    変数が無い comp では KeyError が出るまま (呼び出し側の `collect` が畳む)。"""
    x_var, y_var = access.POTENTIAL_VAR, access.latent_vars(1)[0]

    def trajectory(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        # access.trace は (t, y) を返す。相平面は値のみ使う
        return tuple(access.trace(ds, comp_id, v)[1] for v in (x_var, y_var))  # type: ignore[return-value]

    fig = new_figure()
    ax = fig.subplots()
    o_x, o_y = trajectory(orig_ds)
    s_x, s_y = trajectory(surr_ds)
    # 原系は黒で「正解」の形。alpha を下げて重なりを見やすくする
    ax.plot(
        o_x, o_y, color="black", linewidth=1.2, alpha=0.6, label="Original (Target)"
    )
    ax.plot(
        s_x, s_y, color="crimson", linewidth=1.0, alpha=0.8, label="Surrogate (SINDy)"
    )
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(f"Attractor Comparison (Comp {comp_id})")
    # ランダム電流だと軌道がボヤける → グリッドがあると位置関係を追いやすい
    ax.grid(True, linestyle=":", alpha=0.5)
    place_legend(ax)
    return fig
