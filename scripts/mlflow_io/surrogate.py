"""学習 experiment (`TARGET_EXP`) の**成果物**: surrogate の pickle + spec.json。

答えるのは「その run の surrogate」だけ (`load_surrogate_runs`)。**どの run が居るか・
選べるかは知らない** (→ `runs` module。読込可否の判定に `load_spec` だけ貸す)。
評価 (波形) も知らない。**選んだ run がそのまま run 軸**で、選択を広げも縮めもしない
(hydra の親子は MLflow UI 上の grouping で、比較の単位ではない)。
"""

import tempfile
from functools import cache
from pathlib import Path

import mlflow
import mlflow.artifacts

from neurosurrogate.surrogate.model import (
    SPEC_FILE,
    Surrogate,
    SurrogateSpec,
)
from neurosurrogate.surrogate.runs import SurrogateRuns

from . import logger

_SURR_ARTIFACT_DIR = "surrogate"


def log_surrogate_model(surrogate: Surrogate) -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        surrogate.save(tmp_str)
        mlflow.log_artifacts(tmp_str, artifact_path=_SURR_ARTIFACT_DIR)


@cache
def _load_surrogate_model(run_id: str) -> Surrogate:
    """run_id → surrogate。**run_id ごとに 1 回だけ** DL + unpickle (marimo のセル
    再実行で何度も要求される)。artifact は run に対し不変なので使い回してよい。"""
    logger.debug(f"Loading surrogate from run {run_id}")
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{_SURR_ARTIFACT_DIR}", dst_path=tmp_str
            )
        )
        return Surrogate.load(local)


@cache
def load_spec(run_id: str) -> SurrogateSpec:
    """run の同定情報だけを読む (spec.json のみ DL)。run 一覧は全 run 分これを呼ぶ
    ので、学習成果物の pickle まで落とさない。失敗の握り潰しは呼ぶ側 (`runs`)。"""
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{_SURR_ARTIFACT_DIR}/{SPEC_FILE}", dst_path=tmp_str
            )
        )
        return SurrogateSpec.read(local)


def _load_run_names(run_ids: list[str]) -> tuple[str, ...]:
    """MLflowのrun ID列を一意なrun名列へ変換する。"""
    names = tuple(mlflow.get_run(run_id).info.run_name for run_id in run_ids)
    if None in names or len(set(names)) != len(names):
        raise ValueError(f"学習run名が欠けるか重複 {names}")
    return tuple(str(name) for name in names)


def load_surrogate_runs(run_ids: list[str]) -> SurrogateRuns:
    """MLflowのrun ID列から、一意なrun名を持つsurrogate列をロードする。"""
    return SurrogateRuns(
        tuple(
            (run_name, _load_surrogate_model(run_id))
            for run_id, run_name in zip(run_ids, _load_run_names(run_ids), strict=True)
        )
    )
