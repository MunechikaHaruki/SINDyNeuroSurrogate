"""学習 run の**一覧と選択肢** (`TARGET_EXP`)。

答えるのは「どんな学習 run が居るか」「そのうち今の系列で選べるのはどれか」だけで、
surrogate の中身は `surrogate` module に任せる (読込可否の判定にだけ meta を借りる)。
run 一覧は marimo の run 選択そのものなので、読込不可の run を落とす判断も選んだ系列で
絞る判断もここ = UI は選択の結果を渡すだけで、どの run が選択肢になるかを組み立てない。
"""

from typing import cast

import mlflow
import pandas as pd
from catalog import SERIES
from tqdm import tqdm

from neurosurrogate.sim.run import replaceable
from neurosurrogate.surrogate.meta import SurrogateMeta

from . import TARGET_EXP, logger
from .surrogate import load_meta

ALL_PRESETS = "(すべて)"  # preset で絞らない選択 (dropdown の既定)
# 選択に出す列 = 見分けに要る名前と置換対象の種類、それに選択の実体 (run_id)。
_RUN_COLUMNS = ["tags.mlflow.runName", "comp_type", "run_id"]


def find_selectable_runs(runs_df: pd.DataFrame, series_name: str | None, preset: str):
    """run 表に出す行 = 選んだ系列を**実際に置換できる**代表 run (parent_id 欠損)
    だけを preset で絞ったもの。系列は名前から引く (呼ぶ側はカタログを触らない)。
    置換可否の判定は `sim.run.replaceable` (ドメイン側) が持ち、「1 本でも置換できれば
    出す」という選択の方針だけがここ。`preset=ALL_PRESETS` で preset は絞らない。"""
    if not series_name:
        return runs_df.iloc[:0][_RUN_COLUMNS]
    return runs_df[
        runs_df["meta"].map(lambda m: replaceable(SERIES[series_name], m))
        & ((preset == ALL_PRESETS) | (runs_df["preset"] == preset))
        & runs_df["parent_id"].isna()
    ][_RUN_COLUMNS]


def _safe_meta(run_id: str) -> SurrogateMeta | None:
    """読込不可 (旧形式など) は None にして選択対象から外す。1 件の失敗で
    experiment 全体を見られなくしない。"""
    try:
        return load_meta(run_id)
    except Exception as e:
        logger.debug(f"run {run_id} の meta 読込失敗: {e}")
        return None


def load_runs():
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
    # 親 run id (hydra --multirun の子が持つ mlflow.parentRunId)。親/単発は欠損 (NaN)。
    # 代表判定 (parent_id.isna()) と sweep 兄弟導出 (surrogate.load_sweep_bundles) の
    # 唯一の鍵。
    # .get は親子 run が皆無で列自体が無い実験でも None → 全 NaN 列に落とす。
    runs_df["parent_id"] = runs_df.get("tags.mlflow.parentRunId")
    runs_df = runs_df[
        ["tags.mlflow.runName", "run_id", "start_time", "parent_id"]
        + [c for c in runs_df.columns if "params" in c]
    ]
    # 各 run の同定情報を dataframe 列として付与 (mlflow params に依存せず meta.json
    # から直接読む)。`meta` 列があれば UI は置換互換を replace ドメインの判定関数で
    # 直接効かせられる (互換基準を UI 側に複製しない) → 表示列には含めず絞り込み専用。
    # `comp_type` は置換対象のコンパートメント種類 = モデルペアの左側。
    # 個別 DL バーは抑制し、読込ループ全体を 1 本の進捗バーに集約。
    runs_df["meta"] = [
        _safe_meta(rid) for rid in tqdm(runs_df["run_id"], desc="meta 読込")
    ]
    excluded = int(runs_df["meta"].isna().sum())
    if excluded:
        logger.info(f"surrogate 読込不可の {excluded} 件を選択対象外")
    runs_df = runs_df[runs_df["meta"].notna()].reset_index(drop=True)
    # 全 run が読込不可 = 保存形式の変更で experiment 丸ごと死んでいる。空の
    # dataframe を下流へ流すと UI 構築が意味不明な例外で落ちるのでここで止める。
    if runs_df.empty:
        raise ValueError(
            f"Experiment '{TARGET_EXP}' の {excluded} 件すべてが読込不可 "
            "(保存形式の変更)。再学習が要る: uv run scripts/main.py"
        )
    runs_df["comp_type"] = [m.comp_type.name for m in runs_df["meta"]]
    # 出自の preset は main.py が MLflow param として記録する (surrogate の pickle
    # には入れない)。列名の mlflow 依存はここで吸収し、未記録 run 込みで欠損許容。
    runs_df["preset"] = runs_df.get("params.preset")
    return runs_df
