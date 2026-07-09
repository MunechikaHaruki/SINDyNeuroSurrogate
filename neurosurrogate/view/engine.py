# mypy: ignore-errors

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure


@dataclass
class TraceSpec:
    data: xr.DataArray
    label: str | None = None
    color: str | None = None
    style: str = "-"

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.time.values, self.data.values.squeeze()


@dataclass
class PanelSpec:
    ylabel: str
    traces: list[TraceSpec] = field(default_factory=list)
    xlabel: str | None = None

    def has_legend(self) -> bool:
        return any(tr.label for tr in self.traces)

    def with_xlabel(self, label: str) -> PanelSpec:
        return PanelSpec(ylabel=self.ylabel, traces=self.traces, xlabel=label)


def draw_engine(
    spec: list[PanelSpec],
) -> Figure:
    panels = [*spec[:-1], spec[-1].with_xlabel("Time [ms]")] if spec else spec

    n_rows = len(panels)
    fig = plt.figure()
    axs = fig.subplots(nrows=n_rows, ncols=1, sharex=True)
    if n_rows == 1:
        axs = [axs]

    for ax, p in zip(axs, panels, strict=False):
        for tr in p.traces:
            x, y = tr.xy()
            ax.plot(x, y, label=tr.label, color=tr.color, linestyle=tr.style)
        ax.set_ylabel(p.ylabel)
        if p.has_legend():
            ax.legend()
        if p.xlabel:
            ax.set_xlabel(p.xlabel)

    fig.tight_layout()
    return fig
