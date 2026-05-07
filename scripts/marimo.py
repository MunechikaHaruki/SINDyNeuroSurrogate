import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from scripts.utils.plots import plot_sindy_coefficients

    # ボタンを作成し、変数 'test_btn' に代入
    load_btn = mo.ui.button(
        label="ここをクリック！", value=False, on_click=lambda x: True
    )
    mo.md(f"""
    ### MLflow データ解析
    - **run_idを選択:** {load_btn}
    """)
    return load_btn, mo, plot_sindy_coefficients


@app.cell(hide_code=True)
def _(load_btn, mo):
    # ボタンが押されたときだけデータを読み込む
    from scripts.utils.mlflow_handler import get_runs_df

    with mo.status.spinner(title="MLflowからデータを読み込み中..."):
        if load_btn.value:
            runs_df = get_runs_df()
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
def _(mo, plot_sindy_coefficients, run_selector):
    # モデルの状態を確認するセル
    from scripts.utils.mlflow_handler import get_run_info

    run_ids = run_selector.value["run_id"].tolist()

    model_infos = {}
    for run_id in run_ids:
        run_info = get_run_info(run_id)
        model_infos[run_id] = {}
        model_infos[run_id]["runName"] = run_info["runName"]
        model_infos[run_id]["equations"] = run_info["equations"]
        model_infos[run_id]["dataset"] = run_info["dataset"]
        model_infos[run_id]["sindy_coef"] = plot_sindy_coefficients(
            **run_info["sindy_coef"]
        )

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
def _(mo, run_ids):

    from typing import get_args

    from scripts.utils.builder import CurrentType
    dropdown = mo.ui.dropdown(options=run_ids)
    current_dropdown = mo.ui.dropdown(["train"] + list(get_args(CurrentType)))
    value_slider = mo.ui.slider(start=0,stop=30,step=1)

    first_row = mo.hstack([mo.md("select experiment"), dropdown])
    second_row = mo.hstack([mo.md("choose type"), current_dropdown])
    third_row = mo.hstack([mo.md("value"),value_slider])
    mo.vstack([first_row, second_row,third_row])
    return current_dropdown, dropdown, value_slider


@app.cell(hide_code=True)
def _(current_dropdown, dropdown, mo, model_infos, value_slider):
    mo.stop(
        dropdown.value is None or current_dropdown.value is None,
        "実験を選択してください",
    )

    from scripts.utils.mlflow_handler import load_surrogate_model
    from scripts.utils.builder import build_simulator_config,build_dataset
    from neurosurrogate.calc_engine import unified_simulator
    from neurosurrogate.model import transform_gate
    from neurosurrogate.profiler import calc_dynamic_metrics
    import copy
    surrogate = load_surrogate_model(dropdown.value)

    if current_dropdown.value == "train":
        simulator_config = model_infos[dropdown.value]["dataset"]
    else:
        simulator_config = build_dataset(current_type=current_dropdown.value,value=value_slider.value)
    print(simulator_config)


    def eval_dataset(surrogate_model, dataset_cfg):
        built_cfg=build_simulator_config(dataset_cfg)
        net=built_cfg["net"]
        original_ds = unified_simulator(**built_cfg)
        surr_net = copy.deepcopy(net)

        target_comp_id=0
    
        surr_net["nodes"][target_comp_id]="surr"
        surr_ds = unified_simulator(
            dt = built_cfg["dt"],u=built_cfg["u"],net=surr_net,
            surrogate_model=surrogate_model
        )
        preprocessed_xr = transform_gate(
            surrogate_model.preprocessor, original_ds, target_comp_id=target_comp_id
        )
        return {
            "datasets":{
                "original": original_ds,
                "preprocessed":preprocessed_xr,
                "surrogate":surr_ds,
                "surr_id":target_comp_id},
            "metrics": calc_dynamic_metrics(
                original_ds, surr_ds, target_comp_id, dataset_cfg["dt"]
            ),
        }

    result=eval_dataset(surrogate, simulator_config)

    from scripts.utils.plots import spec_simple,spec_diff,draw_engine
    #draw_engine(spec_simple(result["datasets"]["preprocessed"]))
    draw_engine(spec_diff(**result["datasets"]))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
