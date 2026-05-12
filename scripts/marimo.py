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
    return


@app.cell(hide_code=True)
def _(analysis, base_button, mo):
    # ボタンが押されたときだけデータを読み込む
    with mo.status.spinner(title="MLflowからデータを読み込み中..."):
        if base_button.load_btn.value:
            base_button.setup_mpl()
            run_selector=analysis.get_mlflow_runselector()
        else:
            run_selector=mo.md("push Reload button")
    run_selector
    return (run_selector,)


@app.cell
def _(mo, run_selector):
    run_ids = run_selector.value["run_id"].tolist()
    runid_dropdown = mo.ui.dropdown(options=run_ids,value=run_ids[0])
    mo.md(f"select experiment{runid_dropdown}")
    return run_ids, runid_dropdown


@app.cell(hide_code=True)
def _(analysis, run_ids):
    analysis.get_model_info_ui(run_ids)
    return


@app.cell
def _(
    analysis,
    base_dataset_ui,
    current_dropdown,
    current_ui,
    eval_comp_dropdown,
    mo,
    runid_dropdown,
    surrogate_ui,
):
    mo.stop(
        runid_dropdown.value is None or current_dropdown.value is None,
        "実験を選択してください",
    )
    result=analysis.eval_dataset(
    run_id=runid_dropdown.value,
    current_type=current_dropdown.value,
    current_params=current_ui.value,
    base_dataset_params=base_dataset_ui.value,
    surrogate_list=surrogate_ui.value,
        eval_comp=eval_comp_dropdown.value
    )

    analysis.view_dataset(result)
    return


if __name__ == "__main__":
    app.run()
