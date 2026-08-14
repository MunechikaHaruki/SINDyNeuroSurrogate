"""学習 experiment (`TARGET_EXP`): surrogate の pickle + meta.json を持つ run。

**評価も描画も知らない** — ここが答えるのは「どんな学習 run が居るか」
(`get_runs_df`) と「その run の surrogate / meta」だけ。run 一覧は marimo の run 選択
そのものなので、読込不可の run を落とす判断もここに置く (選択肢に出せない run は
下流へ流さない)。
"""

import json
import tempfile
from functools import cache
from pathlib import Path
from typing import cast

import mlflow
import mlflow.artifacts
import pandas as pd
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from tqdm import tqdm

from neurosurrogate.surrogate.bundle import META_FILE, SurrogateBundle
from neurosurrogate.surrogate.meta import SurrogateMeta

from . import TARGET_EXP, logger

SURR_ARTIFACT_DIR = "surrogate"


def log_surrogate_model(surrogate: SurrogateBundle) -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        surrogate.save(tmp_str)
        mlflow.log_artifacts(tmp_str, artifact_path=SURR_ARTIFACT_DIR)


@cache
def load_surrogate_model(run_id: str) -> SurrogateBundle:
    """run_id → surrogate。**run_id ごとに 1 回だけ** artifact を DL して unpickle
    する。同じ run が一覧走査 (get_runs_df の meta 読込) と選択後のロードで最低
    2 回、marimo のセル再実行のたびに何度も要求されるため。artifact は run に対し
    不変なので、返す bundle を使い回してよい (bundle 側も load 後は書き換えない)。
    """
    logger.debug(f"Loading surrogate from run {run_id}")
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{SURR_ARTIFACT_DIR}", dst_path=tmp_str
            )
        )
        return SurrogateBundle.load(local)


@cache
def load_surrogate_meta(run_id: str) -> SurrogateMeta:
    """run の同定情報だけを読む (meta.json のみ DL)。run 一覧は全 run 分これを呼ぶ
    ので、学習成果物の pickle まで落とさない。"""
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{SURR_ARTIFACT_DIR}/{META_FILE}", dst_path=tmp_str
            )
        )
        return SurrogateMeta.from_dict(json.loads(local.read_text()))


def load_runs(run_ids: list[str]) -> list[SurrogateBundle]:
    """run_id 列 → surrogate ロード。run 選択の唯一のロード経路
    (sweep 複数 / single 1件 共通)。表示名は meta.label (runName 非依存)。"""
    return [load_surrogate_model(rid) for rid in run_ids]


def load_bundles(run_ids: list[str]) -> dict[str, SurrogateBundle]:
    """run_id 列 → run_id→surrogate。他層は表示名でなく **run_id で** surrogate を
    引く (表示名が要る描画層は `report.run_names` で解く)。"""
    return dict(zip(run_ids, load_runs(run_ids), strict=True))


def sweep_siblings(parent_id: str) -> list[str]:
    """親 run (or 単発) の run_id → その sweep 群 = 自身 + 子 (parentRunId 一致)。
    db を直接引く (runs_df 非依存)。run_selector は代表だけ出すので引数は常に親、
    単発は子ゼロ = 1 件。"""
    children = mlflow.search_runs(
        filter_string=f"tags.`{MLFLOW_PARENT_RUN_ID}` = '{parent_id}'",
        output_format="list",
    )
    return [parent_id, *[r.info.run_id for r in children]]


def run_name(run_id: str) -> str:
    """MLflow の run 名。**保存段の名前** = MLflow UI の run 名なので、ディレクトリから
    UI 側の run をそのまま引ける。引けない run (消された等) は id を段名にして描画は
    通す (名前が取れないことは図を出さない理由にならない)。

    学習 run にもレポート run にも効く (run 名は experiment を問わない属性)。"""
    try:
        return mlflow.get_run(run_id).info.run_name or run_id
    except Exception as e:
        logger.debug(f"run {run_id} の名前解決に失敗: {e}")
        return run_id


def _safe_meta(run_id: str) -> SurrogateMeta | None:
    """読込不可 (旧形式など) は None にして選択対象から外す。1 件の失敗で
    experiment 全体を見られなくしない。"""
    try:
        return load_surrogate_meta(run_id)
    except Exception as e:
        logger.debug(f"run {run_id} の meta 読込失敗: {e}")
        return None


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
    # 親 run id (hydra --multirun の子が持つ mlflow.parentRunId)。親/単発は欠損 (NaN)。
    # 代表判定 (parent_id.isna()) と sweep 兄弟導出 (sweep_siblings) の唯一の鍵。
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
