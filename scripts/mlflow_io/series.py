"""波形 experiment (`EVAL_EXP`): **1 run = 1 `EvalSeries`** = 掃引点の波形をまとめた
1 artifact。

中身は 2 種類が並ぶだけ:

    kind=original  … 掃引の原系。surrogate に依存しないので `series_hash` で共有される
    kind=surrogate … その掃引を 1 つの学習 run の surrogate で回したもの

親子関係は張らない。置換系は `original_hash` で自分の原系を名指しするので、同じ原系を
何本の置換系が参照しても run 階層は平坦なまま。点は run の中の並び順そのもの
(点ごとの run も点ごとの識別子も無い)。**どの波形をまとめて 1 回の評価とみなすか**は
この層の関心でない (`report` module が持つ)。

smoke test は本番の評価結果を汚さない別 experiment へ (学習側の MLFLOW_EXPERIMENT と
同じ流儀)。
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

from neurosurrogate.sim.eval import EvalSeries, SimResult

from . import logger

EVAL_EXP = os.environ.get("MLFLOW_EVAL_EXPERIMENT", "eval_series")
WAVES_FILE = "waves.joblib"  # 点の順に並べた波形列 (1 run = 1 系列 = 1 ファイル)
WAVE_DTYPE = "float32"  # 保存精度 (表示にも指標にも十分で容量は半分)
_KIND_ORIGINAL = "original"
_KIND_SURROGATE = "surrogate"


def _eval_exp_id() -> str:
    """評価 experiment の id (無ければ作る)。`set_experiment` は学習側の既定を
    書き換えてしまうので使わず、run ごとに experiment_id を指定する。"""
    exp = mlflow.get_experiment_by_name(EVAL_EXP)
    return exp.experiment_id if exp else mlflow.create_experiment(EVAL_EXP)


def _series_hash(series: EvalSeries, run_id: str | None) -> str:
    """「この掃引をこの surrogate で既に回したか」の鍵。`EvalSeries.hash` は掃引の
    内容だけ (surrogate を含まない) なので、run_id はここで組む = 原系は鍵が掃引だけに
    なり、学習 run を増やしても共有される。"""
    return series.hash() if run_id is None else f"{series.hash()}-{run_id[:8]}"


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


def _log_series(
    name: str,
    series: EvalSeries,
    results: list[SimResult],
    run_id: str | None,
) -> str:
    """1 系列 (点列まるごと) を 1 評価 run へ。`series` param が掃引の単一源で、
    読み戻しはそこからの `EvalSeries.attach` = 点ごとの識別子は保存しない。
    平坦化した param は MLflow UI での絞り込み/比較用の索引。

    run 名は同じ系列の原系と置換系が UI 上で並ぶので kind を添える (置換系はさらに
    どの学習 run のものかを短縮 id で分ける)。**表示名でしかない** — 読み戻しは
    `name` param と tag だけを見る。"""
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
                "n_points": len(results),
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
                    r.dataset.map(lambda v: v.astype(WAVE_DTYPE), keep_attrs=True)
                    for r in results
                ],
                path,
                compress=1,  # float32 波形はほぼ非圧縮 → 高レベルは時間の無駄
            )
            mlflow.log_artifact(str(path))
        logger.info("評価 run 保存: %s [%s] (%s)", name, kind, run.info.run_id)
        return run.info.run_id


def run_series(name: str, series: EvalSeries, run_id: str | None, force: bool) -> str:
    """1 系列 → 波形 run の id。既に同じ掃引 (同じ surrogate) の run があればそれを
    返すだけ = **回さない** (シミュは決定的)。`force=True` は無条件に回し直して新しい
    run を積む。

    **回すか否かと保存は分けない** — 「決定的だから同じ入力は再計算しない」は探索と
    実行が対で成り立つ不変条件で、割ると呼ぶ側が同じ判断を持つことになる。"""
    found = None if force else _find_eval(series, run_id)
    if found is not None:
        return found.info.run_id
    return _log_series(name, series, series.simulate(), run_id)


def source_run_of(eval_run_id: str) -> str:
    """置換系の波形 run → それを回した**学習 run の id** (波形 run 自身が param で
    持つ)。レポート側が学習 run との対応表を別に持たずに済む = レポートは波形 run の
    id だけを指せばよい。"""
    return mlflow.get_run(eval_run_id).data.params["run_id"]


def name_of(eval_run_id: str) -> str:
    """波形 run → 回したときの系列名 (表示用)。**回した記録**であって設定への参照では
    ないので、カタログが変わっても過去の run はそのまま読める。"""
    return mlflow.get_run(eval_run_id).data.params["name"]


def _datasets_of(eval_run_id: str) -> list[xr.Dataset]:
    """評価 run → 点の順に並んだ波形列 (artifact を読む)。"""
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            f"runs:/{eval_run_id}/{WAVES_FILE}", dst_path=tmp
        )
        return cast(list[xr.Dataset], joblib.load(local))


def results_of(eval_run_id: str) -> list[SimResult]:
    """波形 run の id → 点列の結果。掃引の定義が run に載っているので、点の並べ直しも
    点ごとの識別子も要らない (`EvalSeries.attach` が貼る)。"""
    series = EvalSeries.from_dict(
        json.loads(mlflow.get_run(eval_run_id).data.params["series"])
    )
    return series.attach(_datasets_of(eval_run_id))
