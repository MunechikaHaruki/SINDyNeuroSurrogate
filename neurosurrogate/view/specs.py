from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from .engine import PanelSpec, TraceSpec, draw_engine
from .plots import plot_2d_attractor_comparison


def spec_simple(ds: xr.Dataset) -> list[PanelSpec]:
    comp_ids = np.unique(ds.coords["comp_id"].values)
    multi = len(comp_ids) > 1
    spec: list[PanelSpec] = [
        PanelSpec("I_ext", [TraceSpec(ds["I_ext"])]),
    ]

    if "I_internal" in ds:
        spec.append(
            PanelSpec(
                "I_internal",
                [
                    TraceSpec(ds["I_internal"].sel(node_id=i), label=f"Comp {i}")
                    for i in comp_ids
                ],
            )
        )

    spec.append(
        PanelSpec(
            "V(t) [mV]",
            [
                TraceSpec(
                    ds["vars"].sel(gate=False, comp_id=i),
                    label=f"V (Comp {i})" if multi else None,
                )
                for i in comp_ids
            ],
        )
    )

    gate_traces = [
        TraceSpec(
            ds["vars"].sel(gate=True, comp_id=i, variable=v),
            label=f"{v} (Comp {i})",
        )
        for i in comp_ids
        for v in np.unique(
            ds["vars"].sel(gate=True, comp_id=i).coords["variable"].values
        )
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
    v_sel = {"comp_id": surr_id, "variable": "V"}
    latent_vars = [v for v in preprocessed.coords["variable"].values if v != "V"]
    gate_data = original["vars"].sel(comp_id=surr_id, gate=True)

    return [
        PanelSpec("I_ext(t)", [TraceSpec(original["I_ext"], color="gold")]),
        PanelSpec(
            "V [mV]",
            [
                TraceSpec(original["vars"].sel(**v_sel), label="orig V", color="blue"),
                TraceSpec(
                    surrogate["vars"].sel(**v_sel),
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
                        preprocessed["vars"].sel(variable=latent),
                        label=f"target {latent}",
                        color="blue",
                    ),
                    TraceSpec(
                        surrogate["vars"].sel(comp_id=surr_id, variable=latent),
                        label=f"surr {latent}",
                        color="red",
                        style="--",
                    ),
                ],
            )
            for latent in latent_vars
        ],
        PanelSpec(
            "orig gates",
            [
                TraceSpec(gate_data.sel(variable=name), label=name)
                for name in gate_data.coords["variable"].values
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
