"""MLflow I/O = **MLflow を知る唯一の場所**。experiment ごとに 1 module:

- `surrogate` (学習): surrogate の pickle + meta.json を持つ run と、その一覧
- `series` (波形): **1 run = 1 `sim.result.SeriesRun`** = 1 列の波形 1 artifact
- `report` (レポート): **1 run = 1 系列 × N モデル** = `series` の run_id への参照表だけ

どれも「experiment id を解く / 同一性の鍵を組む / 既存を探す / 書く」の 4 点セット。
**再 export はしない** — 呼ぶ側が import 文でどの experiment を触るか名乗る。
ここ (`__init__`) が持つのは tracking 先の固定だけ (import 時に必ず通る)。
"""

import logging
import os
from pathlib import Path

import mlflow

TARGET_EXP = "test_static_params"

logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    """tracking 先をリポジトリ直下の `mlflow.db` に固定する (**import 時に実行**)。
    MLflow 3 の既定 URI は cwd 相対なので、放っておくと居たディレクトリに空 DB が生え
    「run が無い」に見える。`__file__` を resolve して辿る = cwd にも symlink にも
    依存しない。"""
    project_root = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    # smoke test は MLFLOW_EXPERIMENT=smoke_test で本番 experiment を汚さず隔離
    # (just clean-test が丸ごと削除)。既定は本番 experiment のまま。
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", TARGET_EXP))
    # 全 run の meta 読込で artifact DL 進捗バーが大量出力 → 抑制
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"


setup_mlflow()
