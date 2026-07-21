from __future__ import annotations

import marimo as mo
import pandas as pd
from analysis.access import current_of, target_of
from analysis.mode import single as analysis_single
from analysis.mode import sweep as analysis_sweep
from analysis.save.panel import Panel, SaveEntry, entry
from mlflow_io import LoadedRun

from neurosurrogate.metrics.eval import EvalResult
from neurosurrogate.models import MCMODELS
from neurosurrogate.view.utils import current_preview_fig


def view_result(
    loaded_single: LoadedRun | None,
    loaded_sweep: list[LoadedRun],
    base_ui: mo.ui.dictionary,
    res_single: EvalResult | None,
    res_sweep: dict | None,
    draw_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    """single / sweep を縦積みし save エントリを連結 (marimo セルの合成を集約)。"""
    html_single, save_single = view_single(loaded_single, base_ui, res_single, draw_ui)
    html_sweep, save_sweep = view_sweep(loaded_sweep, res_sweep, draw_ui)
    return mo.vstack([html_single, html_sweep]), save_single + save_sweep


def view_single(
    run: LoadedRun | None,
    base_ui: mo.ui.dictionary,
    res: EvalResult | None,
    draw_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    """NeuronGraph → 係数 heatmap (選択 run 静的) → single 波形評価 (res ゲート)。"""
    panel = Panel()
    if run is None:
        panel.note("(single Run 未選択)")
        return panel.done()

    net = MCMODELS[target_of(base_ui)]
    for title, name, fig in analysis_single.model_figs(net, run.surrogate):
        panel.figs(title, [(name, fig)])

    if res is None:
        panel.note("(single結果なし)")
        return panel.done()

    eval_comp = str(draw_ui["eval_comp"].value)
    html, figs, dfs = analysis_single.view_result(draw_ui["single"], res, eval_comp)
    panel.section("波形評価 (single)", html)
    for name, fig in figs:
        panel.save(name, fig)
    panel.save("metrics", dfs["metrics"])
    panel.save("metrics_scalar", dfs["metrics(scalar)"])
    return panel.done()


def _eval_df(loaded: list[LoadedRun]) -> pd.DataFrame:
    rows = [
        {
            "run_name": r.run_name,
            "run_id": r.run_id[:8],
            **r.surrogate.metrics(),
        }
        for r in loaded
    ]
    return pd.DataFrame(rows).set_index("run_name")


def view_sweep(
    loaded: list[LoadedRun],
    res: dict | None,
    draw_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    """先頭に評価サマリ表 (選択 run 静的) → 続けて sweep 結果 (res ゲート)。"""
    panel = Panel()
    if not loaded:
        panel.note("(sweep Run 未選択)")
        return panel.done()

    summary = _eval_df(loaded)
    panel.section("評価サマリ (preprocessor / OpCost)", summary)
    panel.save("eval_summary", summary)

    if res is None or "sweep" not in draw_ui:
        panel.note("(sweep結果なし)")
        return panel.done()

    eval_comp = str(draw_ui["eval_comp"].value)
    ylim_ui = draw_ui["sweep"]["ylim"]
    ylim = (
        None
        if ylim_ui["auto"].value
        else (float(ylim_ui["min"].value), float(ylim_ui["max"].value))
    )
    trace_html, trace_fig = analysis_sweep.plot_sweep_traces(res, eval_comp)
    panel.section("Sweep 波形 (列=amp / 行=各run vs orig)", trace_html)
    panel.save("sweep_traces", trace_fig)

    html, fig = analysis_sweep.plot_sweep(
        res,
        eval_comp_name=eval_comp,
        metric_key=draw_ui["sweep"]["metric"].value,
        ylim=ylim,
    )
    panel.section("Sweep メトリクス", html)
    panel.save("sweep", fig)
    return panel.done()


def plot_preview(
    base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary
) -> tuple[mo.Html, list[SaveEntry]]:
    current_type = current_of(base_ui)
    fig = current_preview_fig(
        current_type,
        float(base_ui["dt"].value),
        setting_ui["sim"]["current_params"].value or {},
    )
    html = mo.vstack([mo.md("### 電流プレビュー"), mo.mpl.interactive(fig)])
    return html, [entry(f"current({current_type})", fig)]
