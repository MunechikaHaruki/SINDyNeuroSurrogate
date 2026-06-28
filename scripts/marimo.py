import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import analysis
    import marimo as mo
    base_button = analysis.make_base_ui()
    analysis.render_base(base_button)
    return analysis, base_button, mo


@app.cell
def _(analysis, base_button):
    analysis.render_model_info(base_button)
    return


@app.cell
def _(analysis, base_button, mo):
    import analysis_sweep

    analysis.setup_mpl(base_button["plt_style"].value)
    sim_ui = analysis.make_sim_ui(base_button, str(base_button["sim_current_type"].value))
    sweep_ui = analysis_sweep.make_sweep_ui(base_button, str(base_button["sweep_current_type"].value))
    mo.hstack(
        [analysis.render_sim_ui(sim_ui), analysis_sweep.render_sweep(sweep_ui)],
        align="start",
    )
    return analysis_sweep, sim_ui, sweep_ui


@app.cell
def _(mo):
    eval_button = mo.ui.run_button(label="単一シミュレーション実行")
    sweep_button = mo.ui.run_button(label="スイープ実行")
    mo.hstack([eval_button, sweep_button])
    return eval_button, sweep_button


@app.cell
def _(analysis, base_button, eval_button, mo, sim_ui):
    mo.stop(not eval_button.value)
    result = analysis.calc_eval(base_button, sim_ui, str(base_button["sim_current_type"].value))
    return (result,)


@app.cell
def _(analysis, base_button):
    draw_ui = analysis.make_draw_ui(base_button)
    return (draw_ui,)


@app.cell
def _(analysis, draw_ui, result):
    spike_ui = analysis.make_spike_ui(result, draw_ui)
    analysis.render_draw_ui(draw_ui, spike_ui)
    return (spike_ui,)


@app.cell
def _(analysis, draw_ui, result, spike_ui):
    html_result, fig_result, dfs_result = analysis.view_result(
        draw_ui, result, spike_ui
    )
    html_result
    return dfs_result, fig_result


@app.cell
def _(analysis_sweep, base_button, mo, sweep_button, sweep_ui):
    mo.stop(not sweep_button.value)
    html_sweep, fig_sweep = analysis_sweep.view_sweep(
        sweep_ui, base_button, str(base_button["sweep_current_type"].value)
    )
    html_sweep
    return (fig_sweep,)


@app.cell
def _(analysis):
    save_defaults = {
        "waveform": "waveform.png",
        "sweep": "sweep.png",
        "neurograph": "neurograph.png",
        "model_info": "model_info.png",
        "waveform_metrics": "waveform_metrics.csv",
        "spike_metrics": "spike_metrics.csv",
        "scalar_metrics": "scalar_metrics.csv",
    }
    save_panel = analysis.make_save_panel(save_defaults)
    analysis.render_save_panel(save_panel, list(save_defaults.keys()))
    return (save_panel,)


@app.cell
def _(analysis, base_button, dfs_result, fig_result, fig_sweep, save_panel):
    analysis.save_panel_items(
        save_panel,
        {
            "waveform": fig_result,
            "sweep": fig_sweep,
            "neurograph": analysis.get_neurograph_fig(base_button),
            "model_info": analysis.get_model_info_figs(base_button),
            "waveform_metrics": dfs_result["waveform_metrics"],
            "spike_metrics": dfs_result["spike_metrics"],
            "scalar_metrics": dfs_result["scalar_metrics"],
        },
    )
    return


if __name__ == "__main__":
    app.run()
