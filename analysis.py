import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import mlflow

    TARGET_EXP="tryDifferentTeachingCurrent"
    mlflow.set_tracking_uri("file:./mlruns")

    # ボタンを作成し、変数 'test_btn' に代入
    load_btn = mo.ui.button(label="ここをクリック！", value=False, on_click=lambda x: True)
    mo.md(f"""
    ### MLflow データ解析
    - **ターゲット実験:** `{TARGET_EXP}`
    - **run_idを選択:** {load_btn}
    """)
    return TARGET_EXP, load_btn, mlflow, mo


@app.cell
def _(TARGET_EXP, load_btn, mlflow, mo):
    # ボタンが押されたときだけデータを読み込む
    with mo.status.spinner(title="MLflowからデータを読み込み中..."):
        if load_btn.value:
            experiment = mlflow.get_experiment_by_name(TARGET_EXP)
            if experiment:
                runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

                # カラムの整理（run_id, start_time を先頭に）
                cols = [c for c in runs_df.columns if "metrics" in c or "params" in c or c == "run_id"]
                runs_df = runs_df[["run_id", "start_time"] + [c for c in cols if c != "run_id"]]

                # selection="multi" に変更
                run_selector = mo.ui.table(
                    runs_df,
                    label="比較・解析したいRunを複数選択してください（Shift/Ctrl+クリック）",
                    selection="multi" 
                )

            else:
                run_selector = mo.md(f"⚠️ 実験 `{TARGET_EXP}` が見つかりませんでした。")
        else:
            run_selector = mo.md("👆 上のボタンを押してデータをロードしてください。")

    run_selector
    return (run_selector,)


@app.cell
def _(run_selector):
    run_ids=run_selector.value["run_id"].tolist()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
