import logging
import os
import tempfile
from pathlib import Path
from typing import cast

import mlflow
import mlflow.artifacts
import pandas as pd
from tqdm import tqdm

from neurosurrogate.surrogate.bundle import SurrogateBundle

TARGET_EXP = "test_static_params"

logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    project_root = Path(__file__).parent.parent
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    # smoke test は MLFLOW_EXPERIMENT=smoke_test で本番 experiment を汚さず隔離
    # (just clean-test が丸ごと削除)。既定は本番 experiment のまま。
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", TARGET_EXP))
    # 全 run の meta 読込で artifact DL 進捗バーが大量出力 → 抑制
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"


SURR_ARTIFACT_DIR = "surrogate"


def log_surrogate_model(surrogate: SurrogateBundle) -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        surrogate.save(tmp_str)
        mlflow.log_artifacts(tmp_str, artifact_path=SURR_ARTIFACT_DIR)


def load_surrogate_model(run_id: str) -> SurrogateBundle:
    logger.debug(f"Loading surrogate from run {run_id}")
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{SURR_ARTIFACT_DIR}", dst_path=tmp_str
            )
        )
        return SurrogateBundle.load(local)


def load_runs(run_ids: list[str]) -> list[SurrogateBundle]:
    """run_id 列 → surrogate ロード。run 選択の唯一のロード経路
    (sweep 複数 / single 1件 共通)。表示名は meta.label (runName 非依存)。"""
    return [load_surrogate_model(rid) for rid in run_ids]


def load_from_selector(selection: pd.DataFrame) -> list[SurrogateBundle]:
    """run_selector の選択 DataFrame → run_id 抽出 → load_runs。
    single (1件) / sweep (複数) 共通の UI ロード経路。"""
    return load_runs(cast(pd.DataFrame, selection)["run_id"].tolist())


def get_runs_df():
    experiment = mlflow.get_experiment_by_name(TARGET_EXP)
    if experiment is None:
        raise ValueError(
            f"Experiment '{TARGET_EXP}' が見つかりません。名前を確認してください。"
        )
    all_runs_df = cast(
        pd.DataFrame, mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    )
    if all_runs_df.empty:
        raise ValueError(f"Experiment '{TARGET_EXP}' にrunが存在しません。")
    runs_df = all_runs_df.copy()
    runs_df = runs_df.sort_values("start_time", ascending=False)
    runs_df["start_time"] = runs_df["start_time"].dt.strftime("%m-%d %H:%M:%S")
    runs_df = runs_df[
        ["tags.mlflow.runName", "run_id", "start_time"]
        + [c for c in runs_df.columns if "metrics" in c or "params" in c]
    ]
    # サロゲートの学習元 MC モデルを純粋な dataframe 列として付与
    # (mlflow params に依存せず meta から直接読む)。個別 DL バーは
    # 抑制し、読込ループ全体を 1 本の進捗バーに集約。
    # 旧形式など読込不可の run は選択対象から除外。
    runs_df["train_model"] = [
        _safe_train_model(rid)
        for rid in tqdm(runs_df["run_id"], desc="surrogate meta 読込")
    ]
    excluded = int(runs_df["train_model"].isna().sum())
    if excluded:
        logger.info(f"surrogate 読込不可の {excluded} 件を選択対象外")
    return runs_df[runs_df["train_model"].notna()].reset_index(drop=True)


def _safe_train_model(run_id: str) -> str | None:
    try:
        return load_surrogate_model(run_id).meta.dataset.model_name
    except Exception:
        return None
