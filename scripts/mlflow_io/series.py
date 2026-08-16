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
from mlflow.entities import Run

from neurosurrogate.sim.result import SeriesRun
from neurosurrogate.sim.run import run_column
from neurosurrogate.sim.spec import EvalSeries
from neurosurrogate.surrogate.bundle import SurrogateBundle

from . import logger

EVAL_EXP = os.environ.get("MLFLOW_EVAL_EXPERIMENT", "eval_series")
WAVES_FILE = "waves.joblib"  # 点の順に並べた波形列 (1 run = 1 系列 = 1 ファイル)
_WAVE_DTYPE = "float32"  # 保存精度 (表示にも指標にも十分で容量は半分)
_KIND_ORIGINAL = "original"
_KIND_SURROGATE = "surrogate"


def _eval_exp_id() -> str:
    """評価 experiment の id (無ければ作る)。`set_experiment` は学習側の既定を
    書き換えるので使わず、run ごとに experiment_id を指定する。"""
    exp = mlflow.get_experiment_by_name(EVAL_EXP)
    return exp.experiment_id if exp else mlflow.create_experiment(EVAL_EXP)


def _series_hash(series: EvalSeries, run_id: str | None) -> str:
    """「この掃引をこの surrogate で既に回したか」の鍵。`EvalSeries.hash` は掃引の内容
    だけなので run_id はここで組む = 原系は鍵が掃引だけになり全レポートで共有される。"""
    return series.hash() if run_id is None else f"{series.hash()}-{run_id}"


def _find_eval(series: EvalSeries, run_id: str | None) -> Run | None:
    """同じ掃引を同じ surrogate で回した評価 run (最新)。決定的なシミュなので、
    あれば回し直す必要はない。"""
    found = mlflow.search_runs(
        experiment_ids=[_eval_exp_id()],
        filter_string=f"tags.series_hash = '{_series_hash(series, run_id)}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    return found[0] if found else None


def _log_series(name: str, column: SeriesRun) -> str:
    """**1 列 = 1 run**。`series` param が掃引の単一源で、波形は同じ並びで置くだけ
    (点ごとの識別子も仕様も保存しない)。平坦化した param は MLflow UI の索引、run 名は
    表示だけ — 読み戻しは `series` / `run_id` と tag しか見ない。"""
    series, run_id = column.series, column.run_id
    kind = _KIND_ORIGINAL if run_id is None else f"{_KIND_SURROGATE}:{run_id[:8]}"
    with mlflow.start_run(
        experiment_id=_eval_exp_id(), run_name=f"{name} [{kind}]"
    ) as run:
        mlflow.log_params(
            {
                "series": json.dumps(series.to_dict(), sort_keys=True, default=str),
                "name": name,
                # MLflow の param は文字列 → None は "None" と書かれて読み戻しで
                # 区別できない。空文字を「無し」の綴りに統一する。
                "run_id": run_id or "",
                "axis": series.param or "",
                "n_points": len(column.waves),
                "target": series.spec.target,
                "current_type": series.spec.current_type,
                "dt": series.spec.dt,
                **{f"cp.{k}": v for k, v in series.spec.current_params.items()},
            }
        )
        mlflow.set_tags(
            {
                "series_hash": _series_hash(series, run_id),
                "kind": _KIND_ORIGINAL if run_id is None else _KIND_SURROGATE,
                **(
                    {}
                    if run_id is None
                    else {"original_hash": _series_hash(series, None)}
                ),
            }
        )
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


def run_series(
    name: str,
    series: EvalSeries,
    run_id: str | None,
    surrogate: SurrogateBundle | None,
    force: bool,
) -> str:
    """1 列 → 波形 run の id。同じ掃引を同じ surrogate で回した run があればそれを返す
    = **回さない** (シミュは決定的。`force` で回し直す)。`run_id`/`surrogate` が両方
    `None` なら原系。**探索と保存は分けない** — 対で成り立つ不変条件なので割らない。"""
    found = None if force else _find_eval(series, run_id)
    if found is not None:
        return found.info.run_id
    return _log_series(name, run_column(series, run_id, surrogate))


def column_of(eval_run_id: str) -> SeriesRun:
    """波形 run の id → **その run が保存した列そのもの** (`_log_series` の逆)。記述も
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
