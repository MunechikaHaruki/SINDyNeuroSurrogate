import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import analysis
    import marimo as mo
    base_button=analysis.get_base_btn()
    base_button.render()
    return analysis, base_button, mo


@app.cell
def _(analysis, base_button):
    param_button=analysis.get_detailed_btn(base_button)
    param_button.render()
    return (param_button,)


@app.cell
def _(base_button, mo):
    run_ids = base_button.run_selector.value["run_id"].tolist()
    runid_dropdown = mo.ui.dropdown(options=run_ids,value=run_ids[0])
    mo.md(f"select experiment{runid_dropdown}")
    return run_ids, runid_dropdown


@app.cell
def _(analysis, mo, param_button):
    eval_comp_dropdown=mo.ui.dropdown(options=param_button.surrogate_target_ui.value,value=param_button.surrogate_target_ui.value[0])
    draw_func_dropdown=mo.ui.dropdown(options=analysis.DRAW_LIST,value=analysis.DRAW_LIST[0])
    mo.md(f"{eval_comp_dropdown},{draw_func_dropdown}")
    return draw_func_dropdown, eval_comp_dropdown


@app.cell(hide_code=True)
def _(analysis, run_ids):
    analysis.get_model_info_ui(run_ids)
    return


@app.cell
def _(analysis, base_button, mo, param_button, runid_dropdown):
    mo.stop(
        runid_dropdown.value is None,
        "実験を選択してください",
    )
    result=analysis.eval_dataset(run_id=runid_dropdown.value,base_btn=base_button,param_ui=param_button)
    return (result,)


@app.cell
def _(analysis, draw_func_dropdown, eval_comp_dropdown, result):

    analysis.view_dataset(result,eval_str=eval_comp_dropdown.value,draw_func_str=draw_func_dropdown.value)
    return


if __name__ == "__main__":
    app.run()
