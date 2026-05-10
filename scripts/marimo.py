import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():

    import analysis
    import marimo as mo

    # ボタンを作成し、変数 'test_btn' に代入
    load_btn = mo.ui.button(
        label="ここをクリック！", value=False, on_click=lambda x: True
    )
    mo.md(f"""
    ### MLflow データ解析
    - **run_idを選択:** {load_btn}
    """)
    return analysis, load_btn, mo


@app.cell(hide_code=True)
def _(analysis, load_btn, mo):
    # ボタンが押されたときだけデータを読み込む

    with mo.status.spinner(title="MLflowからデータを読み込み中..."):
        if load_btn.value:
            runs_df = analysis.get_runs_df()
            if runs_df is None:
                run_selector = mo.md("⚠️ 実験が見つかりませんでした。")
            run_selector = mo.ui.table(
                runs_df[["tags.mlflow.runName", "run_id"]],
                label="比較・解析したいRunを複数選択してください（Shift/Ctrl+クリック）",
                selection="multi",
            )
        else:
            run_selector = mo.md("👆 上のボタンを押してデータをロードしてください。")
    run_selector
    return (run_selector,)


@app.cell(hide_code=True)
def _(analysis, mo, run_selector):
    run_ids = run_selector.value["run_id"].tolist()
    model_infos = analysis.get_model_infos(run_ids)
    mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(
                        f"run_id:{run_id[:8]}.. &nbsp;&nbsp;　{model_infos[run_id]['runName']}"
                    ),
                    mo.md(f"{model_infos[run_id]['equations'][:40]}"),
                    mo.mpl.interactive(model_infos[run_id]["sindy_coef"]),
                ]
            )
            for run_id in run_ids
        ]
    )
    return model_infos, run_ids


@app.cell(hide_code=True)
def _(analysis, mo, run_ids):
    dropdown = mo.ui.dropdown(options=run_ids)
    current_dropdown = mo.ui.dropdown(analysis.CurrentList)
    value_slider = mo.ui.slider(start=0, stop=30, step=1)

    first_row = mo.hstack([mo.md("select experiment"), dropdown])
    second_row = mo.hstack([mo.md("choose type"), current_dropdown])
    third_row = mo.hstack([mo.md("value"), value_slider])
    mo.vstack([first_row, second_row, third_row])
    return current_dropdown, dropdown, value_slider


@app.cell
def _(analysis, dropdown):
    surrogate_model = analysis.load_surrogate_model(dropdown.value)
    print(surrogate_model.surr_comp.gate_names)
    print(type(surrogate_model.surr_comp))
    print(surrogate_model.sindy_args[0].shape)  # xi_matrixのshape

    print(surrogate_model.surr_comp.vars)
    print(surrogate_model.surr_comp.gate)
    print(surrogate_model.surr_comp.init)
    print(analysis.CurrentList)
    return


@app.cell(hide_code=True)
def _(analysis, current_dropdown, dropdown, mo, model_infos, value_slider):
    mo.stop(
        dropdown.value is None or current_dropdown.value is None,
        "実験を選択してください",
    )

    simulator_config=analysis.resolve_config(model_infos,dropdown.value,current_dropdown.value,value_slider.value)
    print(simulator_config)

    result = analysis.eval_dataset(dropdown.value, simulator_config)

    analysis.view_dataset(result)
    return (result,)


@app.cell
def _(result):
    result["metrics"]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
