import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import analysis

    base_button = analysis.make_base_ui()
    analysis.render_base(base_button)
    return analysis, base_button


@app.cell
def _(analysis, base_button):
    analysis.setup_mpl(base_button["plt_style"].value)
    param_button = analysis.make_param_ui(base_button)
    analysis.render_param(param_button)
    return (param_button,)


@app.cell
def _(analysis, param_button):
    eval_ui = analysis.make_eval_ui(param_button)
    analysis.render_eval(eval_ui)
    return (eval_ui,)


@app.cell
def _(analysis, base_button):
    analysis.render_model_info(base_button)
    return


@app.cell
def _(analysis, base_button):
    analysis.render_neurograph(base_button)
    return


@app.cell
def _(analysis, base_button, param_button):
    result = analysis.calc_eval(base_button, param_button)
    return (result,)


@app.cell
def _(analysis, eval_ui, result):
    spike_ui = analysis.make_spike_ui(result, eval_ui)
    analysis.render_spike(spike_ui)
    return (spike_ui,)


@app.cell
def _(analysis, eval_ui, result, spike_ui):
    html_result, fig_result = analysis.view_result(eval_ui, result, spike_ui)
    html_result
    return (fig_result,)


@app.cell
def _(base_button):
    import analysis_sweep

    sweep_ui = analysis_sweep.make_sweep_ui(base_button)
    analysis_sweep.render_sweep(sweep_ui)
    return analysis_sweep, sweep_ui


@app.cell
def _(analysis_sweep, base_button, eval_ui, param_button, sweep_ui):
    html_sweep, fig_sweep = analysis_sweep.view_sweep(
        sweep_ui, base_button, param_button, eval_ui
    )
    html_sweep
    return (fig_sweep,)


@app.cell
def _(analysis):
    save_defaults = {
        "waveform": "waveform.png",
        "sweep": "sweep.png",
    }
    save_panel = analysis.make_save_panel(save_defaults)
    analysis.render_save_panel(save_panel, list(save_defaults.keys()))
    return (save_panel,)


@app.cell
def _(analysis, fig_result, fig_sweep, save_panel):
    analysis.save_panel_figs(
        save_panel,
        {"waveform": fig_result, "sweep": fig_sweep},
    )
    return


if __name__ == "__main__":
    app.run()
