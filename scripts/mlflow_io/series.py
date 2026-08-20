"""波形 experiment (`EVAL_EXP`): **1 run = 1 `sim.result.SeriesRun`** = 1 列の波形を
まとめた 1 artifact。kind=original (原系、`series_hash` で共有) と kind=surrogate が
平坦に並び、置換系は `original_hash` で原系を名指す。点は run の中の並び順そのもの。
**どの列をまとめて 1 回の評価とみなすか**はこの層の関心でない (`report` module)。
"""

import json
import os
import tempfile
from pathlib import Path
from typing import cast

import joblib
import mlflow
import mlflow.artifacts
import xarray as xr

from neurosurrogate.sim.result import SeriesRun
from neurosurrogate.sim.run import run_column
from neurosurrogate.sim.spec import EvalSeries
from neurosurrogate.surrogate.model import Surrogate

from . import logger
from ._query import exp_id, latest_by_tag

EVAL_EXP = os.environ.get("MLFLOW_EVAL_EXPERIMENT", "eval_series")
WAVES_FILE = "waves.joblib"  # 点の順に並べた波形列 (1 run = 1 系列 = 1 ファイル)
_WAVE_DTYPE = "float32"  # 保存精度 (表示にも指標にも十分で容量は半分)
_HASH_TAG = "series_hash"  # 同一性 = 掃引 (+ 置換系なら学習 run_id)
_KIND_ORIGINAL = "original"
_KIND_SURROGATE = "surrogate"


def _series_hash(series: EvalSeries, run_id: str | None) -> str:
    """「この掃引をこの surrogate で既に回したか」の鍵。置換器 (学習 run_id) を組むのは
    ここだけで、掃引側の鍵は記述が持つ: 原系は `hash` (置換範囲を含まない = 置換範囲
    だけが違う対照系列と共有される)、置換系は `replaced_hash` (置換範囲を含む)。"""
    if run_id is None:
        return series.hash()
    return f"{series.replaced_hash()}-{run_id}"


def _tags(series: EvalSeries, run_id: str | None) -> dict[str, str]:
    """波形列の同一性・種類・原系への参照を MLflow tag へ落とす。"""
    return {
        _HASH_TAG: _series_hash(series, run_id),
        "kind": _KIND_ORIGINAL if run_id is None else _KIND_SURROGATE,
        **({} if run_id is None else {"original_hash": _series_hash(series, None)}),
    }


def run_series(
    name: str,
    series: EvalSeries,
    run_id: str | None,
    surrogate: Surrogate | None,
) -> str:
    """1 列 → 波形 run の id。同じ掃引を同じ surrogate で回した run があればそれを返す
    = **回さない** (シミュは決定的なので、鍵が一致した run は常に正しい)。
    `run_id`/`surrogate` が両方 `None` なら原系。**探索と保存は分けない** — 対で
    成り立つ不変条件なので割らない。"""
    # 同じ掃引を同じ surrogate で回した run があるか (決定的なシミュ = 回し直さない)。
    found = latest_by_tag(EVAL_EXP, _HASH_TAG, _series_hash(series, run_id))
    if found is not None:
        return found.info.run_id

    column = run_column(series, run_id, surrogate)
    kind = _KIND_ORIGINAL if run_id is None else f"{_KIND_SURROGATE}:{run_id[:8]}"
    with mlflow.start_run(
        experiment_id=exp_id(EVAL_EXP), run_name=f"{name} [{kind}]"
    ) as run:
        mlflow.log_params(
            {
                "series": json.dumps(series.to_dict(), sort_keys=True, default=str),
                # MLflow の param は文字列 → None は "None" と書かれて読み戻しで
                # 区別できない。空文字を「無し」の綴りに統一する。
                "run_id": run_id or "",
            }
        )
        mlflow.set_tags(_tags(series, run_id))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / WAVES_FILE
            joblib.dump(
                [
                    ds.map(lambda v: v.astype(_WAVE_DTYPE), keep_attrs=True)
                    for ds in column.waves
                ],
                path,
                compress=1,  # float32 波形はほぼ非圧縮 → 高レベルは時間の無駄
            )
            mlflow.log_artifact(str(path))
        logger.info("評価 run 保存: %s [%s] (%s)", name, kind, run.info.run_id)
        return run.info.run_id


def load_column(eval_run_id: str) -> SeriesRun:
    """波形 run の id → **その run が保存した列そのもの** (`run_series` の逆)。記述も
    run_id も param が持つので、呼ぶ側は id 1 つを指すだけでよい。"""
    params = mlflow.get_run(eval_run_id).data.params
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            f"runs:/{eval_run_id}/{WAVES_FILE}", dst_path=tmp
        )
        return SeriesRun(
            EvalSeries.from_dict(json.loads(params["series"])),
            # 空文字が「無し」の綴り (MLflow の param は文字列なので None を書けない)。
            params["run_id"] or None,
            cast(list[xr.Dataset], joblib.load(local)),
        )
