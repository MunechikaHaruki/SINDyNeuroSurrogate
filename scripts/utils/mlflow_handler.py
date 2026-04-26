import tempfile
from pathlib import Path

import mlflow

TARGET_EXP = "test_static_params"
mlflow.set_tracking_uri("file:./mlruns")


def get_runs_df():
    experiment = mlflow.get_experiment_by_name(TARGET_EXP)
    if experiment:
        all_runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        # 親runのみを抽出
        runs_df = all_runs_df[all_runs_df["tags.mlflow.parentRunId"].isna()].copy()
        # start_time に基づいて降順（最新が上）にソート、表示変更
        runs_df = runs_df.sort_values("start_time", ascending=False)
        runs_df["start_time"] = runs_df["start_time"].dt.strftime("%m-%d %H:%M:%S")

        # カラムの整理（run_id, start_time を先頭に）
        cols = [
            c
            for c in runs_df.columns
            if "metrics" in c or "params" in c or c == "run_id"
        ]
        runs_df = runs_df[
            ["tags.mlflow.runName", "run_id", "start_time"]
            + [c for c in cols if c != "run_id"]
        ]
        return runs_df
    else:
        return None


def get_model_informations(run_ids):
    client = mlflow.MlflowClient()
    artifact_path = "sindy_coef.png"
    download_dir = Path(tempfile.mkdtemp())
    model_info = {}
    for run_id in run_ids:
        model_info[run_id] = {}
        dest = download_dir / run_id
        dest.mkdir(exist_ok=True)

        local_path = client.download_artifacts(
            run_id=run_id, path=artifact_path, dst_path=str(dest)
        )

        model_info[run_id]["sindy_coef"] = local_path
        model_info[run_id]["runName"] = client.get_run(run_id).data.tags[
            "mlflow.runName"
        ]
        model_info[run_id]["equations"] = mlflow.artifacts.load_text(
            f"runs:/{run_id}/equations.txt"
        )
        model_info[run_id]["teaching_config"] = mlflow.artifacts.load_text(
            f"runs:/{run_id}/dataset.yaml"
        )
    return model_info


def get_child_runs(parent_run_ids):
    # 親Runのexperiment_idを取得
    experiment_id = mlflow.get_run(parent_run_ids[0]).info.experiment_id
    # 全Runを取得してからPythonで子Runをフィルタリング
    all_runs_df = mlflow.search_runs(experiment_ids=[experiment_id])

    # 親Runに紐付く子Runをフィルタリング
    child_runs_df = all_runs_df[
        all_runs_df["tags.mlflow.parentRunId"].isin(parent_run_ids)
    ].reset_index(drop=True)
    return child_runs_df


# # 親が選択されていない場合は停止
# mo.stop(len(run_selector.value) == 0)
# with mo.status.spinner(title="子Runを取得中..."):
#     child_runs_df=get_child_runs(run_selector.value["run_id"].tolist())
# # 子Runを一覧表示（ここから特定の評価結果を選ぶ）
# if len(child_runs_df) > 0:
#     child_selector = mo.ui.table(
#         child_runs_df[["tags.mlflow.runName", "run_id", "tags.eval_dataset", "start_time"]],
#         label="Artifactを確認したい子Runを選択してください",
#         selection="single"
#     )
# else:
#     mo.md("**子Runが見つかりません**")

# child_selector

# # 子Runが選択されていない場合は停止
# import mlflow
# mo.stop(len(child_selector.value) == 0)

# target_run_id = child_selector.value.iloc[0]["run_id"]

# with mo.status.spinner(title="Artifactリストを取得中..."):
#     client = mlflow.tracking.MlflowClient()
#     # Artifactのリスト（ファイル一覧）を取得
#     artifacts = client.list_artifacts(target_run_id)

#     # ファイル名の一覧を作成
#     artifact_paths = [a.path for a in artifacts]

# # Artifactを選択するドロップダウン
# artifact_selector = mo.ui.dropdown(
#     options=artifact_paths,
#     label="表示するArtifactを選択:"
# )

# artifact_selector

# mo.stop(not artifact_selector.value)

# path = artifact_selector.value
# local_path = client.download_artifacts(target_run_id, path)

# if path.endswith((".png", ".jpg", ".jpeg")):
#     # 画像の場合
#     content = mo.image(local_path)
# elif path.endswith((".txt", ".yaml", ".json", ".log")):
#     # テキスト系の場合
#     with open(local_path, "r") as f:
#         content = mo.plain_text(f.read())
# else:
#     content = mo.md(f"📁 ファイルをダウンロードしました: `{local_path}`")

# mo.vstack([
#     mo.md(f"### Artifact: {path}"),
#     content
# ])

# import matplotlib.pyplot as plt
# import numpy as np
# import yaml

# from scripts.flow import build_current_pipeline

# # yaml読み込み
# with open("./scripts/conf/config.yaml") as f:
#     cfg = yaml.safe_load(f)

# # UI
# selected = mo.ui.dropdown(
#     options=list(cfg["current_train_pipelines"].keys()), label="experiment"
# )
# mo.hstack([selected])

# # 選択されたexpの電流を生成して表示
# pipeline = cfg["current_train_pipelines"][selected.value]

# current_cfg = {"pipeline": pipeline}

# defaults = cfg["datasets_default"]
# dt = defaults["simulator_default_dt"]
# iteration = int(defaults["simulator_default_duration"] / dt)
# current_cfg.setdefault("current_seed", defaults["default_current_seed"])
# current_cfg.setdefault("iteration", iteration)
# current_cfg.setdefault("silence_steps", int(defaults["silence_duration"] / dt))

# # パイプライン実行
# u = build_current_pipeline(current_cfg)
# t = np.arange(iteration) * dt

# fig, ax = plt.subplots()
# ax.plot(t, u)
# ax.set_xlabel("time [ms]")
# ax.set_ylabel("I_ext")
# mo.mpl.interactive(fig)
