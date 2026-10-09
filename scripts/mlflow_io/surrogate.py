"""学習 experiment (`_TARGET_EXP`) の**成果物**: surrogate の pickle + spec.json と、
学習 run 1 本が自分について描ける図 (`figures/`)。

答えるのは「その run の surrogate」だけ (`load_surrogate_runs`)。**どの run が居るか・
選べるかは知らない** (選ぶのは live-textbook の冊)。評価 (波形) も知らない。
**選んだ run がそのまま run 軸**で、選択を広げも縮めもしない (hydra の親子は
MLflow UI 上の grouping で、比較の単位ではない)。
"""

import tempfile
from functools import cache
from pathlib import Path

import mlflow
import mlflow.artifacts

from neurosurrogate.artifact.plotting import use_style
from neurosurrogate.surrogate.artifacts import surrogate_artifacts
from neurosurrogate.surrogate.model import Surrogate
from neurosurrogate.surrogate.runs import SurrogateRuns

from . import logger

_SURR_ARTIFACT_DIR = "surrogate"
_FIGURES_DIR = "figures"


def log_surrogate_model(surrogate: Surrogate) -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        surrogate.save(tmp_str)
        mlflow.log_artifacts(tmp_str, artifact_path=_SURR_ARTIFACT_DIR)


def log_surrogate_figures(surrogate: Surrogate) -> None:
    """学習 run 1 本が自分について描ける図を、開いている run の `figures/` へ置く。
    学習 run だけで決まる図なので、学習のときに一度だけ描けば足りる。"""
    use_style()
    with tempfile.TemporaryDirectory() as tmp_str:
        surrogate_artifacts(surrogate).save(Path(tmp_str))
        mlflow.log_artifacts(tmp_str, artifact_path=_FIGURES_DIR)


@cache
def _load_surrogate_model(run_id: str) -> Surrogate:
    """run_id → surrogate。**run_id ごとに 1 回だけ** DL + unpickle。artifact は run に
    対し不変なので使い回してよい。"""
    logger.debug(f"Loading surrogate from run {run_id}")
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{_SURR_ARTIFACT_DIR}", dst_path=tmp_str
            )
        )
        return Surrogate.load(local)


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
