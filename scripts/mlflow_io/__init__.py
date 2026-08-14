"""MLflow I/O = **MLflow を知る唯一の場所**。experiment ごとに 1 module。

- `surrogate` (学習): surrogate の pickle + meta.json を artifact に持つ run と、その
  一覧 (`get_runs_df`)。
- `series` (波形): **1 run = 1 `EvalSeries`** = 掃引点の波形をまとめた 1 artifact。
  原系の run (`kind=original`) と置換系の run (`kind=surrogate`) がフラットに並び、
  置換系は `tags.original_hash` で自分の原系を名指す。原系は掃引の内容だけで同一性が
  決まるので、学習 run を増やしても複製されない。
- `report` (レポート): **1 run = 1 レポート = 1 系列 × N モデル** = 「どの学習 run 群を
  どの系列で回したか」。持つのは `series` の run_id への**参照表だけ**で、波形の実体は
  複製しない (原系は複数レポートで共有される資産)。描画はこの単位を読む。

3 つとも「experiment id を解く / 同一性の鍵を組む / 既存を探す / 書く」の同じ 4 点
セットで出来ており、その 1 組が 1 module に収まる。**再 export はしない** — 呼ぶ側は
どの experiment を触っているかを import 文で名乗る
(`from mlflow_io.report import ...`)。

再実行はシミュが決定的 (Euler、乱数なし) なことを使って `tags.series_hash` 一致で
スキップする。`force=True` のときだけ回し直して新しい run を積む。

ここ (package の `__init__`) が持つのは **tracking 先の固定だけ**。どの module を
import しても最初に通るので、URI を張り忘れようがない。
"""

import logging
import os
from pathlib import Path

import mlflow

TARGET_EXP = "test_static_params"

logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    """tracking 先をリポジトリ直下の `mlflow.db` に固定する (**import 時に実行**)。

    MLflow 3 の既定 tracking URI は **cwd 相対**の `sqlite:///mlflow.db` → 設定前に
    この package の関数を呼ぶと、そのとき居たディレクトリに空 DB が生えて「run が無い」
    に見える。URI を持つのはこの package なので、呼び忘れようのない import 時に張る
    (`__file__` は resolve してから辿る = cwd にも symlink にも依存しない)。
    """
    project_root = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    # smoke test は MLFLOW_EXPERIMENT=smoke_test で本番 experiment を汚さず隔離
    # (just clean-test が丸ごと削除)。既定は本番 experiment のまま。
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", TARGET_EXP))
    # 全 run の meta 読込で artifact DL 進捗バーが大量出力 → 抑制
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"


setup_mlflow()
