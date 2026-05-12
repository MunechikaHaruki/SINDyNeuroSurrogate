import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import analysis
    import analysis_sweep
    import marimo as mo
    base_button = analysis.BaseUI.get_base_btn()
    base_button.render()
    return analysis, analysis_sweep, base_button, mo


@app.cell
def _(analysis, base_button):
    base_button.setup_mpl()
    param_button = analysis.ParamUI.get_detailed_btn(base_button)
    param_button.render()
    return (param_button,)


@app.cell
def _(analysis, param_button):
    eval_ui=analysis.EvalUI.get_eval_ui(param_button)
    eval_ui.render()
    return (eval_ui,)


@app.cell
def _(base_button):
    base_button.get_model_info_ui()
    return


@app.cell
def _(analysis, base_button, param_button):

    result = analysis.eval_dataset(base_btn=base_button, param_ui=param_button)
    return (result,)


@app.cell
def _(eval_ui, result):
    eval_ui.view_result(result)
    return


@app.cell
def _(analysis_sweep, base_button):

    sweep_ui = analysis_sweep.SweepUI.build(base_button)
    sweep_ui.render()
    return (sweep_ui,)


@app.cell
def _(base_button, mo, sweep_ui):
    fig=sweep_ui.run_and_plot(base_button,base_button.run_ids)
    mo.mpl.interactive(fig)
    return


if __name__ == "__main__":
    app.run()
