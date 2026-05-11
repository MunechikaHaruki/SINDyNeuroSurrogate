import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import analysis
    import marimo as mo
    ui,load_btn,plt_btn,current_dropdown=analysis.init_cell()
    ui
    return analysis, current_dropdown, load_btn, mo, plt_btn


@app.cell
def _(analysis, current_dropdown):
    current_param_ui=analysis.get_param_ui(current_dropdown.value)
    current_param_ui
    return (current_param_ui,)


@app.cell(hide_code=True)
def _(analysis, load_btn, mo, plt_btn):
    # ボタンが押されたときだけデータを読み込む
    with mo.status.spinner(title="MLflowからデータを読み込み中..."):
        if load_btn.value:
            analysis.setup_matplotlib(plt_btn.value)
            run_selector=analysis.get_mlflow_runselector()
        else:
            run_selector=mo.md("push Reload button")
    run_selector
    return (run_selector,)


@app.cell
def _(mo, run_ids):
    mo.stop(run_ids is None,"Choose Run")
    runid_dropdown = mo.ui.dropdown(options=run_ids,value=run_ids[0])
    mo.hstack([mo.md("select experiment"), runid_dropdown])
    return (runid_dropdown,)


@app.cell(hide_code=True)
def _(analysis, run_selector):
    run_ids = run_selector.value["run_id"].tolist()
    analysis.get_model_info_ui(run_ids)
    return (run_ids,)


@app.cell
def _(analysis, current_dropdown, current_param_ui, mo, runid_dropdown):
    mo.stop(
        runid_dropdown.value is None or current_dropdown.value is None,
        "実験を選択してください",
    )
    result=analysis.eval_dataset(runid_dropdown.value,current_dropdown.value,current_param_ui.value)

    analysis.view_dataset(result)
    return


if __name__ == "__main__":
    app.run()
