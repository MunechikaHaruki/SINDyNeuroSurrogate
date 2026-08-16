"""レポート experiment (`REPORT_EXP`): **1 run = 1 レポート = 1 系列 × N モデル**。

持つのは波形 run の id 2 tag だけで、波形もカタログ由来の値も持たない。同一性は選択
そのもの (`sim.spec.EvalSelection.hash`) で、同じ選択で回し直すと参照だけ差し替わる
→ param でなく tag。marimo の評価ボタンが `run_and_log`、描画ボタンが `render_report`。
"""

import json
import os
from pathlib import Path

import mlflow
from mlflow.entities import Run
from tuning import Tuning

from neurosurrogate.artifact.bundle import (
    detail_artifacts,
    original_artifacts,
    report_artifacts,
    surrogate_artifacts,
)
from neurosurrogate.artifact.model import Artifacts
from neurosurrogate.sim.result import SeriesResults
from neurosurrogate.sim.run import replaced_runs
from neurosurrogate.sim.spec import EvalSelection, EvalSeries
from neurosurrogate.surrogate.bundle import SurrogateBundle

from . import logger
from .save import per_run, save_artifacts
from .series import column_of, run_series
from .surrogate import load_bundles

REPORT_EXP = os.environ.get("MLFLOW_REPORT_EXPERIMENT", "eval_report")
_ORIGINAL_TAG = "original_series_id"
_SURROGATE_TAG = "surrogate_series_ids"  # 波形 run id の列 (与えた順 = 凡例の並び)


def _report_exp_id() -> str:
    """書く側だけが呼ぶ (無ければ作る)。読む側は `_report_exp_id_if_any`。"""
    exp = mlflow.get_experiment_by_name(REPORT_EXP)
    return exp.experiment_id if exp else mlflow.create_experiment(REPORT_EXP)


def _report_exp_id_if_any() -> str | None:
    """探すだけの経路は experiment を作らない (選択を変えるたび空の experiment が
    生えない)。"""
    exp = mlflow.get_experiment_by_name(REPORT_EXP)
    return exp.experiment_id if exp else None


def _find_report(report_hash: str) -> Run | None:
    exp_id = _report_exp_id_if_any()
    if exp_id is None:
        return None
    found = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.report_hash = '{report_hash}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    return found[0] if found else None


def _log_report(name: str, original: str, surrs: list[str], report_hash: str) -> str:
    """波形 run の id 2 つを 1 レポート run へ。同じ選択の run があれば**参照先だけ
    差し替える** → param でなく tag。`name` は表示にだけ使う。"""
    # client 直で書く: 既存 run への追記を active experiment に左右されないため。
    client = mlflow.MlflowClient()
    found = _find_report(report_hash)
    run_id = found.info.run_id if found else _new_report_run(client, name, report_hash)
    client.set_tag(run_id, _ORIGINAL_TAG, original)
    # **与えた順を保つ** (sort しない): 選択順が凡例/行見出しの並びとして
    # 描画層まで効く。
    client.set_tag(run_id, _SURROGATE_TAG, json.dumps(surrs))
    logger.info("レポート run 保存: %s %s (%s)", name, report_hash, run_id)
    return run_id


def _new_report_run(client: mlflow.MlflowClient, name: str, report_hash: str) -> str:
    """空のレポート run を 1 本立てる (同一性の tag と表示名まで)。参照先は呼ぶ側。"""
    run = client.create_run(
        _report_exp_id(),
        tags={
            "report_hash": report_hash,
            "mlflow.runName": f"{name} [{report_hash}]",
        },
    )
    client.set_terminated(run.info.run_id)
    return run.info.run_id


def find_report_run(selection: EvalSelection) -> str | None:
    """選択 → 既存レポート run の id (無ければ None)。**選択からレポート run_id へ渡す
    唯一の橋**で、以降 (描画) はこの id 1 つだけを見る。"""
    found = _find_report(selection.hash())
    return found.info.run_id if found else None


def run_and_log(
    bundles: dict[str, SurrogateBundle],
    name: str,
    series: EvalSeries,
    force: bool = False,
) -> str:
    """1 系列の評価実行 + 波形 run 保存 + レポート run 保存 (marimo の評価ボタン)。
    既にある波形 run は再利用 = 回さない。返りはレポート run の id = そのまま描画の
    入力。1 本も置換できない選択は回す意味が無いので拒む。"""
    surrs = replaced_runs(series, bundles)
    if not surrs:
        raise ValueError(f"{name}: 選択 run のどれでも置換できない (比較対象が無い)")
    return _log_report(
        name,
        run_series(name, series, None, None, force),
        [run_series(name, series, rid, bundle, force) for rid, bundle in surrs.items()],
        EvalSelection(series, tuple(bundles)).hash(),
    )


def load_report(report_run_id: str) -> SeriesResults:
    """**レポート run_id 1 つ** → 描く中身。掃引の記述も学習 run との対応も波形 run 側
    から戻る (`series.column_of`) = カタログにも学習 experiment にも依存しない。
    **返すのは描く中身だけ** — 読んだ波形 run の id は描画にも保存段にも要らない。"""
    tags = mlflow.get_run(report_run_id).data.tags
    surrs: list[str] = json.loads(tags[_SURROGATE_TAG])
    return SeriesResults(
        column_of(tags[_ORIGINAL_TAG]), tuple(column_of(eid) for eid in surrs)
    )


def _series_artifacts(
    view: SeriesResults, bundles: dict[str, SurrogateBundle], tuning: Tuning
) -> dict[Path, Artifacts]:
    """波形 1 本ずつで決まる図 (`series/<run 名>/`) = 原系の入力電流と置換系の詳細図。
    段名は学習 run の MLflow run 名 (`models/` と同じ綴りで引ける)。"""
    return {
        Path("series/original"): original_artifacts(view),
        **per_run(
            "series",
            {
                run_id: detail_artifacts(
                    view,
                    run_id,
                    bundles[run_id],
                    tuning.eval_comp,
                    tuning.view_comps,
                    tuning.detail_point,
                    tuning.spike_orig,
                    tuning.spike_surr,
                )
                for run_id in view.run_ids
            },
        ),
    }


def render_report(report_run_id: str, tuning: Tuning) -> list[str]:
    """レポート run を描画し、全成果物を同じ run へ保存する唯一の interface。呼ぶ側が
    知るのは id と描画条件だけで、参照解決も surrogate ロードも区分もここに閉じる。"""
    view = load_report(report_run_id)
    bundles = load_bundles(view.run_ids)
    directories = per_run(
        "models",
        {
            run_id: surrogate_artifacts(bundle, tuning.view_comps)
            for run_id, bundle in bundles.items()
        },
    )
    directories.update(_series_artifacts(view, bundles, tuning))
    directories[Path()] = report_artifacts(
        view,
        bundles,
        tuning.eval_comp,
        tuning.metric,
        tuning.metric_ylim,
    )
    return save_artifacts(
        directories,
        report_run_id,
        tuning,
    )
