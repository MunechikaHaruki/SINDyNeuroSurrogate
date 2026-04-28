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
    from scripts.utils.mlflow_handler import (
        get_model_informations,
        get_runs_df,
    )

    # ボタンを作成し、変数 'test_btn' に代入
    load_btn = mo.ui.button(
        label="ここをクリック！", value=False, on_click=lambda x: True
    )
    mo.md(f"""
    ### MLflow データ解析
    - **run_idを選択:** {load_btn}
    """)
    return get_model_informations, get_runs_df, load_btn, mo


@app.cell(hide_code=True)
def _(TARGET_EXP, get_runs_df, load_btn, mo):
    # ボタンが押されたときだけデータを読み込む

    with mo.status.spinner(title="MLflowからデータを読み込み中..."):
        if load_btn.value:
            runs_df = get_runs_df()
            if runs_df is None:
                run_selector = mo.md(f"⚠️ 実験 `{TARGET_EXP}` が見つかりませんでした。")
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
def _(get_model_informations, mo, run_selector):
    # モデルの状態を確認するセル
    run_ids = run_selector.value["run_id"].tolist()
    model_infos = get_model_informations(run_ids)
    mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(
                        f"run_id:{run_id[:8]}.. &nbsp;&nbsp;　{model_infos[run_id]['runName']}"
                    ),
                    mo.md(f"{model_infos[run_id]['equations'][:40]}"),
                    mo.image(src=model_infos[run_id]["sindy_coef"]),
                ]
            )
            for run_id in run_ids
        ]
    )
    return (run_ids,)


@app.cell(hide_code=True)
def _(mo, run_ids):
    dropdown = mo.ui.dropdown(options=run_ids)
    from typing import get_args

    from scripts.utils.builder import CurrentType

    current_dropdown = mo.ui.dropdown(["train"] + list(get_args(CurrentType)))

    first_row = mo.hstack([mo.md("select experiment"), dropdown])
    second_row = mo.hstack([mo.md("choose type"), current_dropdown])
    mo.vstack([first_row, second_row])
    return current_dropdown, dropdown


@app.cell
def _(current_dropdown, dropdown, mo):
    mo.stop(
        dropdown.value is None or current_dropdown.value is None,
        "実験を選択してください",
    )
    print(dropdown.value)
    print(current_dropdown.value)
    from scripts.utils.mlflow_handler import load_surrogate_model

    surrogate = load_surrogate_model(dropdown.value)
    # simulator_config = model_infos[dropdown.value]["teaching_config"]
    # eval_dataset(surrogate,simulator_config)

    # type: random
    # catalog:
    #   random: [9919, 9920]
    #   steady: {start: 0, stop: 30, step: 1 }
    #   sweep:
    #     model_name: hh
    #     duration: 800
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
