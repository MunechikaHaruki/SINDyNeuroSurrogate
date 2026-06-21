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
    dataset_cfg, run_id, surrogate_targets = analysis.to_eval_params(
        base_button, param_button
    )
    result = analysis.build_eval_result(dataset_cfg, run_id, surrogate_targets)
    return (result,)


@app.cell
def _(analysis, eval_ui, result):
    spike_ui = analysis.make_spike_ui(result, eval_ui)
    analysis.render_spike(spike_ui)
    return (spike_ui,)


@app.cell
def _(analysis, eval_ui, result, spike_ui):
    analysis.view_result(eval_ui, result, spike_ui)
    return


@app.cell
def _(base_button, eval_ui, mo, param_button):
    import analysis_sweep
    run_ids = base_button["run_selector"].value["run_id"].tolist()
    sweep_df, sweep_fig = analysis_sweep.run_and_plot(param_button["sweep_ui"], base_button, param_button, eval_ui, run_ids)
    mo.vstack([mo.mpl.interactive(sweep_fig), mo.ui.table(sweep_df)])
    return


if __name__ == "__main__":
    app.run()
