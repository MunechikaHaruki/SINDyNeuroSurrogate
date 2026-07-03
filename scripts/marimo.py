import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import analysis
    import marimo as mo

    base_button = analysis.make_base_ui()
    base_button
    return analysis, base_button, mo


@app.cell
def _(analysis, base_button, mo):
    analysis.setup_mpl(base_button["plt_style"].value)
    combined_ui = analysis.make_combined_ui(base_button)
    run_button = mo.ui.run_button(label="実行")
    mo.vstack(
        [
            analysis.render_combined_ui(combined_ui),
            run_button,
        ]
    )
    return combined_ui, run_button


@app.cell
def _(analysis, base_button, combined_ui):
    fig_current = analysis.plot_current_preview(base_button, combined_ui["sim"])
    analysis.render_current_preview(fig_current)
    return (fig_current,)


@app.cell
def _(analysis, base_button):
    draw_ui = analysis.make_draw_ui(base_button)
    analysis.render_draw_ui(draw_ui)
    return (draw_ui,)


@app.cell
def _(analysis, draw_ui, result):
    spike_ui = analysis.make_spike_ui(result, draw_ui)
    analysis.render_spike_ui(spike_ui)
    return (spike_ui,)


@app.cell
def _(analysis, draw_ui, mo, result, spike_ui, sweep_result):
    html_result, fig_result, dfs_result = analysis.view_result(
        draw_ui, result, spike_ui
    )
    html_sweep, fig_sweep = analysis.plot_sweep(sweep_result)
    mo.vstack([html_result, html_sweep])
    return dfs_result, fig_result, fig_sweep


@app.cell
def _(analysis, base_button):
    model_info_figs = analysis.get_model_info_figs(base_button)
    save_defaults = {
        "waveform": "_waveform.png",
        "sweep": "_sweep.png",
        "current_preview": "_current_preview.png",
        "neurograph": "_neurograph.png",
        **{f"model_info_{k}": f"_model_info_{k}.png" for k in model_info_figs},
        "metrics": "_metrics.csv",
        "scalar_metrics": "_scalar_metrics.csv",
    }
    save_panel = analysis.make_save_panel(save_defaults)
    analysis.render_save_panel(save_panel)
    return model_info_figs, save_panel


@app.cell
def _(
    analysis,
    base_button,
    dfs_result,
    fig_current,
    fig_result,
    fig_sweep,
    model_info_figs,
    save_panel,
):
    analysis.save_panel_items(
        save_panel,
        {
            "waveform": fig_result,
            "sweep": fig_sweep,
            "current_preview": fig_current,
            "neurograph": analysis.get_neurograph_fig(base_button),
            **{f"model_info_{k}": v for k, v in model_info_figs.items()},
            "metrics": dfs_result["metrics"],
            "scalar_metrics": dfs_result["scalar_metrics"],
        },
    )
    return


@app.cell(column=1)
def _(analysis, base_button):
    analysis.render_model_info(base_button)
    return


@app.cell
def _(analysis, base_button, combined_ui, draw_ui, mo, run_button):
    mo.stop(not run_button.value)
    surrogate_targets = combined_ui["surrogate_targets"].value
    result = analysis.calc_eval(base_button, combined_ui["sim"], surrogate_targets)
    sweep_result = analysis.calc_sweep(
        base_button, combined_ui["sweep"], surrogate_targets, draw_ui
    )
    return result, sweep_result


if __name__ == "__main__":
    app.run()
