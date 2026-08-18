"""学習 experiment (`TARGET_EXP`) の**成果物**: surrogate の pickle + meta.json。

答えるのは「その run の surrogate」だけ (`load_bundles` / `load_sweep_bundles`) で、
**どの run が居るか・選べるかは知らない** (→ `runs` module。読込可否の判定に
`load_meta` だけ貸す)。
評価 (波形) も知らない。選択 (代表 N 件) と run 軸 (sweep 兄弟込み M 件) の差は
`load_sweep_bundles` だけが吸収する = UI は選んだ代表を渡すだけ。
"""

import json
import tempfile
from functools import cache
from pathlib import Path

import mlflow
import mlflow.artifacts
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from neurosurrogate.surrogate.bundle import META_FILE, SurrogateBundle
from neurosurrogate.surrogate.meta import SurrogateMeta

from . import logger

_SURR_ARTIFACT_DIR = "surrogate"


def log_surrogate_model(surrogate: SurrogateBundle) -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        surrogate.save(tmp_str)
        mlflow.log_artifacts(tmp_str, artifact_path=_SURR_ARTIFACT_DIR)


@cache
def _load_surrogate_model(run_id: str) -> SurrogateBundle:
    """run_id → surrogate。**run_id ごとに 1 回だけ** DL + unpickle (marimo のセル
    再実行で何度も要求される)。artifact は run に対し不変なので使い回してよい。"""
    logger.debug(f"Loading surrogate from run {run_id}")
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{_SURR_ARTIFACT_DIR}", dst_path=tmp_str
            )
        )
        return SurrogateBundle.load(local)


@cache
def load_meta(run_id: str) -> SurrogateMeta:
    """run の同定情報だけを読む (meta.json のみ DL)。run 一覧は全 run 分これを呼ぶ
    ので、学習成果物の pickle まで落とさない。失敗の握り潰しは呼ぶ側 (`runs`)。"""
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{_SURR_ARTIFACT_DIR}/{META_FILE}", dst_path=tmp_str
            )
        )
        return SurrogateMeta.from_dict(json.loads(local.read_text()))


def load_bundles(run_ids: list[str]) -> dict[str, SurrogateBundle]:
    """run_id 列 → run_id→surrogate。他層は表示名でなく **run_id で** surrogate を
    引く (表示名が要る描画層は `sim.artifacts` 側で解く)。"""
    return {run_id: _load_surrogate_model(run_id) for run_id in run_ids}


def _sweep_siblings(parent_id: str) -> list[str]:
    """親 run (or 単発) の run_id → sweep 群 = 自身 + 子 (parentRunId 一致)。db を直接
    引く (runs 一覧に非依存)。選択肢は代表だけなので引数は常に親。"""
    children = mlflow.search_runs(
        filter_string=f"tags.`{MLFLOW_PARENT_RUN_ID}` = '{parent_id}'",
        output_format="list",
    )
    return [parent_id, *[r.info.run_id for r in children]]


def load_sweep_bundles(selected_ids: list[str]) -> dict[str, SurrogateBundle]:
    """選んだ代表 run 列 → run_id→surrogate。**選択 (代表 N 件) と run 軸 (兄弟込み
    M 件) の差はここだけが知る**: 各代表の hydra sweep 兄弟へ与えた順で広げ、選択が
    重なっても同じ run が 2 度出ないよう潰す。単発 preset は 1 件のまま。"""
    return load_bundles(
        list(dict.fromkeys(rid for sel in selected_ids for rid in _sweep_siblings(sel)))
    )
