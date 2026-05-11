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
    param_ui=analysis.get_param_ui(current_dropdown.value)
    param_ui
    return (param_ui,)


@app.cell(hide_code=True)
def _(analysis, load_btn, mo, plt_btn):
    # ボタンが押されたときだけデータを読み込む
    analysis.setup_matplotlib(plt_btn.value)
    with mo.status.spinner(title="MLflowからデータを読み込み中..."):
        if load_btn.value:
            run_selector=analysis.get_mlflow_runselector()
        else:
            run_selector=mo.md("mlflowをReloadして")
    run_selector
    return (run_selector,)


@app.cell
def _(mo, run_ids):
    dropdown = mo.ui.dropdown(options=run_ids,value=run_ids[0])
    mo.hstack([mo.md("select experiment"), dropdown])
    return (dropdown,)


@app.cell(hide_code=True)
def _(analysis, run_selector):
    run_ids = run_selector.value["run_id"].tolist()
    analysis.get_model_info_ui(run_ids)
    return (run_ids,)


@app.cell(hide_code=True)
def _(analysis, current_dropdown, dropdown, mo, param_ui):
    mo.stop(
        dropdown.value is None or current_dropdown.value is None,
        "実験を選択してください",
    )

    simulator_config=analysis.resolve_config(dropdown.value,current_dropdown.value,param_ui.value)
    print(simulator_config)

    result = analysis.eval_dataset(dropdown.value, simulator_config)

    analysis.view_dataset(result)
    return (result,)


@app.cell
def _(mo, result):
    metrics=result["metrics"]

    mo.hstack([
        mo.stat(label=k, value=f"{v:.4f}" if isinstance(v, float) else str(v))
        for k, v in metrics.items()
    ])
    return


if __name__ == "__main__":
    app.run()
