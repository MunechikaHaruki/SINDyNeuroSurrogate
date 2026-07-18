from __future__ import annotations

import marimo as mo
import pandas as pd
from analysis.mode import single as analysis_single
from analysis.mode import sweep as analysis_sweep
from analysis.panel import Panel, SaveEntry, entry, pair
from analysis.ui import target_of
from mlflow_io import LoadedRun

from neurosurrogate.metrics.eval import EvalResult
from neurosurrogate.models import MCMODELS
from neurosurrogate.surrogate.replace import replaced_names
from neurosurrogate.view.model import view_model, view_neuron_graph
from neurosurrogate.view.utils import current_preview_fig


def view_single(
    run: LoadedRun | None,
    base_ui: mo.ui.dictionary,
    res: EvalResult | None,
    draw_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    """NeuronGraph → 係数 heatmap (選択 run 静的) → single 波形評価 (res ゲート)。"""
    pair_tag = pair(base_ui)
    panel = Panel()
    if run is None:
        panel.note("(single Run 未選択)")
        return panel.done()

    net = MCMODELS[target_of(base_ui)]
    surr = run.surrogate
    panel.figs(
        "NeuronGraph",
        [
            (
                f"neurograph({pair_tag})",
                view_neuron_graph(net, replaced_names(surr, net)),
            )
        ],
    )
    panel.figs("SINDy 係数", [(f"model({pair_tag})", view_model(surr.sindy_bundle))])

    if res is None:
        panel.note("(single結果なし)")
        return panel.done()

    eval_comp = str(draw_ui["eval_comp"].value)
    suffix = f"{pair_tag},{eval_comp}"
    html, figs, dfs = analysis_single.view_result(draw_ui["single"], res, eval_comp)
    panel.section("波形評価 (single)", html)
    for name, fig in figs:
        panel.save(f"{name}({suffix})", fig)
    panel.save(f"metrics({suffix})", dfs["metrics"])
    panel.save(f"metrics_scalar({suffix})", dfs["metrics(scalar)"])
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
    base_ui: mo.ui.dictionary,
    res: dict | None,
    draw_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    """先頭に評価サマリ表 (選択 run 静的) → 続けて sweep 結果 (res ゲート)。"""
    pair_tag = pair(base_ui)
    summary = _eval_df(loaded)
    panel = Panel()
    panel.section("評価サマリ (preprocessor / OpCost)", summary)
    panel.save(f"eval_summary({pair_tag})", summary)

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
    panel.save(f"sweep_traces({pair_tag},{eval_comp})", trace_fig)

    html, fig = analysis_sweep.plot_sweep(
        res,
        eval_comp_name=eval_comp,
        metric_key=draw_ui["sweep"]["metric"].value,
        ylim=ylim,
    )
    panel.section("Sweep メトリクス", html)
    panel.save(f"sweep({pair_tag},{eval_comp})", fig)
    return panel.done()


def plot_preview(
    base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary
) -> tuple[mo.Html, list[SaveEntry]]:
    current_type = str(base_ui["sim_current_type"].value)
    fig = current_preview_fig(
        current_type,
        float(base_ui["dt"].value),
        setting_ui["sim"]["current_params"].value or {},
    )
    html = mo.vstack([mo.md("### 電流プレビュー"), mo.mpl.interactive(fig)])
    return html, [entry(f"current({current_type})", fig)]
