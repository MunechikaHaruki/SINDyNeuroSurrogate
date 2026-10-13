"""MLflow I/O = **experiment と run を知る唯一の場所**。experiment ごとに 1 module:

- `surrogate` (学習): surrogate の pickle + spec.json と、中身 (`description/`)
- `_series` (波形): **1 run = 1 `sim.result.SeriesRun`** = 1 列の波形
- `evaluation`: 1 系列を原系と選んだ学習 run で回す (`just eval` の中身)

どれも「experiment id を解く / 同一性の鍵を組む / 既存を探す / 書く」の 4 点セット。
公開名の綴りも揃える: `run_*` = 確保 (無ければ回して書き、あれば既存 run の id)、
`load_*` = 読み (run_id / 選択 → 中身)。`find_*` はそこから外れる問い合わせ専用
(回さない)。`log_*` = active run への書き足しだけ (同一性を見ない = 確保ではない)。
図は描かない (描くのは figures/ の図)。書く形は `_files` が持つ。
**再 export はしない** — 呼ぶ側が import 文でどの experiment を触るか名乗る。
ここ (`__init__`) が持つのは tracking 先の固定だけ (import 時に必ず通る)。
"""

import logging
import os
from pathlib import Path

import mlflow

_TARGET_EXP = "test_static_params"

logger = logging.getLogger(__name__)


def _setup_mlflow() -> None:
    """tracking 先をリポジトリ直下の `mlflow.db` に固定する (**import 時に実行**)。
    MLflow 3 の既定 URI は cwd 相対なので、放っておくと居たディレクトリに空 DB が生え
    「run が無い」に見える。`__file__` を resolve して辿る = cwd にも symlink にも
    依存しない。"""
    project_root = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    # smoke test は MLFLOW_EXPERIMENT=smoke_test で本番 experiment を汚さず隔離
    # (just clean-test が丸ごと削除)。既定は本番 experiment のまま。
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", _TARGET_EXP))
    # 全 run の spec 読込で artifact DL 進捗バーが大量出力 → 抑制
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"


_setup_mlflow()
