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
    analysis.setup_mpl(base_button["plt_style"].value)
    combined_ui = analysis.make_combined_ui(base_button)
    run_button = mo.ui.run_button(label="実行")
    mo.vstack([
        analysis.render_combined_ui(combined_ui),
        run_button,
    ])
    return combined_ui, run_button


@app.cell
def _(analysis, base_button):
    draw_ui = analysis.make_draw_ui(base_button)
    analysis.render_draw_ui(draw_ui)
    return (draw_ui,)


@app.cell
def _(analysis, base_button, combined_ui, draw_ui, mo, run_button):
    mo.stop(not run_button.value)
    surrogate_targets = combined_ui["surrogate_targets"].value
    result = analysis.calc_eval(base_button, combined_ui["sim"], surrogate_targets)
    sweep_result = analysis.calc_sweep(
        base_button, combined_ui["sweep"], surrogate_targets, draw_ui
    )
    return result, sweep_result


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
