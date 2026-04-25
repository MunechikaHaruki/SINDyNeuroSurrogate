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
