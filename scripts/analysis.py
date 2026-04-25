import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
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
                all_runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                # 親runのみを抽出
                runs_df = all_runs_df[all_runs_df["tags.mlflow.parentRunId"].isna()].copy()
                # start_time に基づいて降順（最新が上）にソート、表示変更
                runs_df = runs_df.sort_values("start_time", ascending=False)
                runs_df["start_time"] = runs_df["start_time"].dt.strftime("%m-%d %H:%M:%S")

                # カラムの整理（run_id, start_time を先頭に）
                cols = [c for c in runs_df.columns if "metrics" in c or "params" in c or c == "run_id"]
                runs_df = runs_df[["tags.mlflow.runName","run_id", "start_time"] + [c for c in cols if c != "run_id"]]

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
    return experiment, run_selector


@app.cell
def _(experiment, mlflow, mo, run_selector):
    # 親が選択されていない場合は停止
    mo.stop(len(run_selector.value) == 0)

    # 選択された親IDのリスト
    parent_run_ids = run_selector.value["run_id"].tolist()

    # フィルタクエリを作成 (tags.mlflow.parentRunId が親IDのいずれかと一致するもの)
    filter_query = " OR ".join([f"tags.mlflow.parentRunId = '{pid}'" for pid in parent_run_ids])

    with mo.status.spinner(title="子Runを取得中..."):
        # 子Run（評価結果など）を検索
        child_runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_query
        )

    # 子Runを一覧表示（ここから特定の評価結果を選ぶ）
    child_selector = mo.ui.table(
        child_runs_df[["tags.mlflow.runName", "run_id", "tags.eval_dataset", "start_time"]],
        label="Artifactを確認したい子Runを選択してください",
        selection="single"
    )

    child_selector
    return (child_selector,)


@app.cell
def _(child_selector, mlflow, mo):
    # 子Runが選択されていない場合は停止
    mo.stop(len(child_selector.value) == 0)

    target_run_id = child_selector.value.iloc[0]["run_id"]

    with mo.status.spinner(title="Artifactリストを取得中..."):
        client = mlflow.tracking.MlflowClient()
        # Artifactのリスト（ファイル一覧）を取得
        artifacts = client.list_artifacts(target_run_id)

        # ファイル名の一覧を作成
        artifact_paths = [a.path for a in artifacts]

    # Artifactを選択するドロップダウン
    artifact_selector = mo.ui.dropdown(
        options=artifact_paths,
        label="表示するArtifactを選択:"
    )

    artifact_selector
    return artifact_selector, client, target_run_id


@app.cell
def _(artifact_selector, client, mo, target_run_id):
    mo.stop(not artifact_selector.value)

    path = artifact_selector.value
    local_path = client.download_artifacts(target_run_id, path)

    if path.endswith((".png", ".jpg", ".jpeg")):
        # 画像の場合
        content = mo.image(local_path)
    elif path.endswith((".txt", ".yaml", ".json", ".log")):
        # テキスト系の場合
        with open(local_path, "r") as f:
            content = mo.plain_text(f.read())
    else:
        content = mo.md(f"📁 ファイルをダウンロードしました: `{local_path}`")

    mo.vstack([
        mo.md(f"### Artifact: {path}"),
        content
    ])
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    from scripts.flow import build_current_pipeline

    # yaml読み込み
    with open("./scripts/conf/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # UI
    selected = mo.ui.dropdown(
        options=list(cfg["current_train_pipelines"].keys()), label="experiment"
    )
    mo.hstack([selected])
    return build_current_pipeline, cfg, mo, np, plt, selected


@app.cell
def _(build_current_pipeline, cfg, mo, np, plt, selected):
    # 選択されたexpの電流を生成して表示
    pipeline = cfg["current_train_pipelines"][selected.value]

    current_cfg = {"pipeline": pipeline}

    defaults = cfg["datasets_default"]
    dt = defaults["simulator_default_dt"]
    iteration = int(defaults["simulator_default_duration"] / dt)
    current_cfg.setdefault("current_seed", defaults["default_current_seed"])
    current_cfg.setdefault("iteration", iteration)
    current_cfg.setdefault("silence_steps", int(defaults["silence_duration"] / dt))

    # パイプライン実行
    u = build_current_pipeline(current_cfg)
    t = np.arange(iteration) * dt

    fig, ax = plt.subplots()
    ax.plot(t, u)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("I_ext")
    mo.mpl.interactive(fig)
    return


if __name__ == "__main__":
    app.run()
