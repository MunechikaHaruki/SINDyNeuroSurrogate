from __future__ import annotations

import marimo as mo
from analysis.access import current_of
from analysis.mode import single as analysis_single
from analysis.mode import sweep as analysis_sweep
from analysis.save.panel import SaveEntry, entry
from mlflow_io import LoadedRun

from neurosurrogate.metrics.eval import EvalResult
from neurosurrogate.view.utils import current_preview_fig


def view_result(
    loaded_single: LoadedRun | None,
    loaded_sweep: list[LoadedRun],
    base_ui: mo.ui.dictionary,
    res_single: EvalResult | None,
    res_sweep: dict | None,
    draw_ui: mo.ui.dictionary,
) -> list[SaveEntry]:
    """single / sweep の save entry 列を連結 (表示は panel.render)。"""
    return analysis_single.view(
        loaded_single, base_ui, res_single, draw_ui
    ) + analysis_sweep.view(loaded_sweep, res_sweep, draw_ui)


def plot_preview(
    base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary
) -> list[SaveEntry]:
    current_type = current_of(base_ui)
    fig = current_preview_fig(
        current_type,
        float(base_ui["dt"].value),
        setting_ui["sim"]["current_params"].value or {},
    )
    return [entry(f"current({current_type})", fig)]
