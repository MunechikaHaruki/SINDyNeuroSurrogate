import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from scripts.utils.mlflow_handler import TARGET_EXP,get_runs_df,get_child_runs,get_model_informations
    from scripts.utils.builder_core import build_simulator_config

    # ボタンを作成し、変数 'test_btn' に代入
    load_btn = mo.ui.button(label="ここをクリック！", value=False, on_click=lambda x: True)
    mo.md(f"""
    ### MLflow データ解析
    - **ターゲット実験:** `{TARGET_EXP}`
    - **run_idを選択:** {load_btn}
    """)
    return (
        TARGET_EXP,
        build_simulator_config,
        get_model_informations,
        get_runs_df,
        load_btn,
        mo,
    )


@app.cell
def _(TARGET_EXP, get_runs_df, load_btn, mo):
    # ボタンが押されたときだけデータを読み込む

    with mo.status.spinner(title="MLflowからデータを読み込み中..."):
        if load_btn.value:
            runs_df=get_runs_df()
            if runs_df is None:
                run_selector = mo.md(f"⚠️ 実験 `{TARGET_EXP}` が見つかりませんでした。")
            run_selector = mo.ui.table(
                runs_df[["tags.mlflow.runName","run_id"]],
                label="比較・解析したいRunを複数選択してください（Shift/Ctrl+クリック）",
                selection="multi" 
            )
        else:
            run_selector = mo.md("👆 上のボタンを押してデータをロードしてください。")
    run_selector
    return (run_selector,)


@app.cell
def _(get_model_informations, mo, run_selector):
    # モデルの状態を確認するセル
    run_ids=run_selector.value["run_id"].tolist()
    model_infos=get_model_informations(run_ids)
    mo.vstack( [
        mo.vstack([
            mo.md(f"run_id:{run_id[:8]}.. &nbsp;&nbsp;　runName:{model_infos[run_id]["runName"]}"),
            mo.md(f"{model_infos[run_id]["equations"][:40]}"),
            mo.image(src=model_infos[run_id]["sindy_coef"])
        ])
        for run_id in run_ids
    ])
    return model_infos, run_ids


@app.cell
def _(mo, run_ids):
    dropdown=mo.ui.dropdown(options=run_ids)
    dropdown
    return (dropdown,)


@app.cell
def _(build_simulator_config, dropdown, model_infos):
    from neurosurrogate.modeling.calc_engine import unified_simulator
    simulator_config=model_infos[dropdown.value]["teaching_config"]
    unified_simulator(**build_simulator_config(simulator_config))
    return


@app.cell
def _(run_ids):
    from scripts.utils.log_model import load_surrogate_model
    surrogate=load_surrogate_model(run_ids[0])
    return


app._unparsable_cell(
    r"""
    print(surrogate.)
    """,
    name="_"
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
