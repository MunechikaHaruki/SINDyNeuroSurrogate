import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import analysis
    import analysis_core
    import analysis_sweep
    import marimo as mo

    base_button = analysis.make_base_ui()
    analysis.render_base(base_button)
    return analysis, analysis_core, analysis_sweep, base_button, mo


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
def _(base_button, mo):
    from neurosurrogate.model.registry_neuron import MCMODELS
    from neurosurrogate.profiler.profiler_view import view_neuron_graph

    _model_name = base_button["base_dataset"].value["model_name"]
    mo.vstack(
        [
            mo.md(f"### NeuronGraph: `{_model_name}`"),
            mo.mpl.interactive(view_neuron_graph(MCMODELS[_model_name])),
        ]
    )
    return


@app.cell
def _(analysis_core, base_button, param_button):
    current_params_val = param_button["current_params"].value
    result = analysis_core.build_eval_result(
        analysis_core.EvalInput(
            current_type=base_button["current_type"].value,
            run_id=param_button["run_id"].value,
            current_params=current_params_val if current_params_val else None,
            base_dataset_params=base_button["base_dataset"].value,
            surrogate_targets=param_button["surrogate_targets"].value,
        )
    )
    return (result,)


@app.cell
def _(analysis, eval_ui, result):
    analysis.view_result(eval_ui, result)
    return


@app.cell
def _(analysis_sweep, base_button):
    sweep_ui = analysis_sweep.make_sweep_ui(base_button)
    analysis_sweep.render_sweep(sweep_ui)
    return (sweep_ui,)


@app.cell
def _(analysis_sweep, base_button, mo, sweep_ui):
    run_ids = base_button["run_selector"].value["run_id"].tolist()
    fig = analysis_sweep.run_and_plot(sweep_ui, base_button, run_ids)
    mo.mpl.interactive(fig)
    return


if __name__ == "__main__":
    app.run()
