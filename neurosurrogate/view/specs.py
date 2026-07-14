from __future__ import annotations

from collections.abc import Callable

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


DrawFn = Callable[[xr.Dataset, xr.Dataset, Callable[[], xr.Dataset], int], Figure]
DRAW_MAP: dict = {
    "diff": lambda orig, surr, get_pre, comp_id: draw_engine(
        spec_diff(orig, get_pre(), surr, surr_id=comp_id)
    ),
    "simple": lambda orig, surr, get_pre, comp_id: draw_engine(spec_simple(orig)),
    "attractor": lambda orig, surr, get_pre, comp_id: plot_2d_attractor_comparison(
        get_pre(), surr, comp_id
    ),
}
