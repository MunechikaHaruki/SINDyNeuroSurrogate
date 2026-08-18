"""レポート experiment (`REPORT_EXP`): **1 評価 = 1 run**。

持つのは波形 run の id 2 tag だけで、波形もカタログ由来の値も持たない。**同じ選択で
回し直したら新しい run が立つ** = 一度書いた run を後から書き換えない (同一性の鍵を
持たない)。重い波形は `series` 側が内容で再利用するので、作り直しの代償は run 1 本の
メタデータだけ。marimo のレポートボタンは `write_report` だけを呼ぶ。

成果物の内容と段構成は `artifact.bundle.save_report` が決め、このmoduleはレポートrun
の参照解決と書き出しを担う。
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from catalog import SERIES

from neurosurrogate.artifact.bundle import save_report
from neurosurrogate.sim.result import SeriesResults

from . import logger
from ._query import exp_id
from .series import load_column, run_series
from .surrogate import load_surrogate_runs

REPORT_EXP = os.environ.get("MLFLOW_REPORT_EXPERIMENT", "eval_report")
_ORIGINAL_TAG = "original_series_id"
_SURROGATE_TAG = "surrogate_series_ids"  # 波形 run id の列 (与えた順 = 凡例の並び)


def write_report(run_ids: tuple[str, ...], name: str, tuning: dict[str, Any]) -> None:
    """設定に基づくレポートを 1 本完成させる、この module 唯一の interface。

    選んだ学習 run と系列を評価し、波形 run を再利用または保存して、新しいレポート
    run に `tuning` どおりの全成果物を書く。
    """
    series = SERIES[name]
    runs = load_surrogate_runs(list(run_ids))
    surrs = runs.replacing(series)
    if not surrs:
        raise ValueError(f"{name}: 選択 run のどれでも置換できない (比較対象が無い)")

    original_id = run_series(name, series, None, None)
    surrogate_ids = [
        run_series(name, series, run_id, bundle)
        for run_id, (run_name, bundle) in zip(run_ids, runs, strict=True)
        if run_name in surrs.names
    ]
    with tempfile.TemporaryDirectory() as temporary:
        save_report(
            SeriesResults(
                load_column(original_id),
                tuple(load_column(run_id) for run_id in surrogate_ids),
            ),
            surrs,
            tuning,
            Path(temporary),
        )
        client = mlflow.MlflowClient()
        report_run_id = client.create_run(
            exp_id(REPORT_EXP),
            tags={
                _ORIGINAL_TAG: original_id,
                # **与えた順を保つ** (sort しない): 選択順が凡例/行見出しの並びとして
                # 描画層まで効く。
                _SURROGATE_TAG: json.dumps(surrogate_ids),
                "mlflow.runName": f"{name} ×{len(surrogate_ids)}",
            },
        ).info.run_id
        try:
            client.log_artifacts(report_run_id, temporary)
        except Exception:
            client.set_terminated(report_run_id, "FAILED")
            raise
        client.set_terminated(report_run_id)
    logger.info("レポート run 保存: %s (%s)", name, report_run_id)
