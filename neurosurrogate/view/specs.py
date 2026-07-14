from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import xarray as xr
from matplotlib.figure import Figure

from ..core import access
from ..core.access import POTENTIAL_VAR
from .engine import PanelSpec, TraceSpec, draw_engine
from .plots import plot_2d_attractor_comparison


def spec_simple(ds: xr.Dataset) -> list[PanelSpec]:
    comp_ids = access.comp_ids(ds)
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
            "V(t) [mV]",
            [
                TraceSpec(
                    *access.trace(ds, i, POTENTIAL_VAR),
                    label=f"V (Comp {i})" if multi else None,
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


def spec_diff(
    original: xr.Dataset,
    preprocessed: xr.Dataset,
    surrogate: xr.Dataset,
    surr_id: int,
) -> list[PanelSpec]:
    return [
        PanelSpec("I_ext(t)", [TraceSpec(*access.i_ext(original), color="gold")]),
        PanelSpec(
            "V [mV]",
            [
                TraceSpec(
                    *access.trace(original, surr_id, POTENTIAL_VAR),
                    label="orig V",
                    color="blue",
                ),
                TraceSpec(
                    *access.trace(surrogate, surr_id, POTENTIAL_VAR),
                    label="surr V",
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
                        *access.trace(preprocessed, surr_id, latent),
                        label=f"target {latent}",
                        color="blue",
                    ),
                    TraceSpec(
                        *access.trace(surrogate, surr_id, latent),
                        label=f"surr {latent}",
                        color="red",
                        style="--",
                    ),
                ],
            )
            for latent in access.latent_variables(preprocessed)
        ],
        PanelSpec(
            "orig gates",
            [
                TraceSpec(*access.trace(original, surr_id, name), label=name)
                for name in access.gate_variables(original, surr_id)
            ],
        ),
    ]


@dataclass(frozen=True)
class PlotContext:
    """描画 registry の共通入力。各 draw 関数は必要なフィールドだけ使う。

    get_preprocessed は lazy: 学習ドメイン外の comp では verdict が raise する
    ため、latent を参照する diff/attractor でのみ評価する (simple は呼ばない)。
    """

    original: xr.Dataset
    surrogate: xr.Dataset
    comp_id: int
    get_preprocessed: Callable[[], xr.Dataset]


def draw_diff(ctx: PlotContext) -> Figure:
    return draw_engine(
        spec_diff(ctx.original, ctx.get_preprocessed(), ctx.surrogate, ctx.comp_id)
    )


def draw_simple(ctx: PlotContext) -> Figure:
    return draw_engine(spec_simple(ctx.original))


def draw_attractor(ctx: PlotContext) -> Figure:
    return plot_2d_attractor_comparison(
        ctx.get_preprocessed(), ctx.surrogate, ctx.comp_id
    )


DRAW_MAP: dict[str, Callable[[PlotContext], Figure]] = {
    "diff": draw_diff,
    "simple": draw_simple,
    "attractor": draw_attractor,
}
