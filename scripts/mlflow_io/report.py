"""レポート experiment (`REPORT_EXP`): **1 評価 = 1 run**。

持つのは波形 run の id 2 tag だけで、波形もカタログ由来の値も持たない。**同じ選択で
回し直したら新しい run が立つ** = 一度書いた run を後から書き換えない (同一性の鍵を
持たない)。重い波形は `series` 側が内容で再利用するので、作り直しの代償は run 1 本の
メタデータだけ。marimo のレポートボタンが `run_report` → `log_report_artifacts`。

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


def _log_report(name: str, original: str, surrs: list[str]) -> str:
    """波形 run の id 2 つを持つレポート run を**1 本新しく立てる**。参照は生成時に
    書き切って以降触らない = この run は立った時点で完成している。"""
    # client 直で書く: 立てる先を active experiment に左右されないため。
    client = mlflow.MlflowClient()
    run = client.create_run(
        exp_id(REPORT_EXP),
        tags={
            _ORIGINAL_TAG: original,
            # **与えた順を保つ** (sort しない): 選択順が凡例/行見出しの並びとして
            # 描画層まで効く。
            _SURROGATE_TAG: json.dumps(surrs),
            "mlflow.runName": f"{name} ×{len(surrs)}",
        },
    )
    client.set_terminated(run.info.run_id)
    logger.info("レポート run 保存: %s (%s)", name, run.info.run_id)
    return run.info.run_id


def run_report(run_ids: tuple[str, ...], name: str) -> str:
    """1 系列の評価実行 + 波形 run 保存 + レポート run 保存 (レポートボタンの前半)。
    系列は**名前から引く** = 呼ぶ側はカタログを触らない。既にある波形 run は再利用
    = 回さない (押すたびに増えるのはレポート run だけ)。返りはレポート run の id =
    そのまま描画の入力。1 本も置換できない選択は回す意味が無いので拒む。"""
    series = SERIES[name]
    runs = load_surrogate_runs(list(run_ids))
    surrs = runs.replacing(series)
    if not surrs:
        raise ValueError(f"{name}: 選択 run のどれでも置換できない (比較対象が無い)")
    return _log_report(
        name,
        run_series(name, series, None, None),
        [
            run_series(name, series, run_id, bundle)
            for run_id, (run_name, bundle) in zip(run_ids, runs, strict=True)
            if run_name in surrs.names
        ],
    )


def load_report(report_run_id: str) -> SeriesResults:
    """**レポート run_id 1 つ** → 描く中身。掃引の記述も学習 run との対応も波形 run 側
    から戻る (`series.load_column`) = カタログにも学習 experiment にも依存しない。
    **返すのは描く中身だけ** — 読んだ波形 run の id は描画にも保存段にも要らない。"""
    tags = mlflow.get_run(report_run_id).data.tags
    surrs: list[str] = json.loads(tags[_SURROGATE_TAG])
    return SeriesResults(
        load_column(tags[_ORIGINAL_TAG]), tuple(load_column(eid) for eid in surrs)
    )


def log_report_artifacts(report_run_id: str, tuning: dict[str, Any]) -> list[str]:
    """レポートrunを描画し、全成果物を同じrunへ書き足す唯一のinterface。

    `tuning` は marimo のつまみそのもの (意味を解くのは `save_report` 1 箇所)。
    そのときの表示設定を `tuning.json` 1枚添える。返りは書いたartifact path列。
    描き直しで同じpathは置き換わり、生成しなかった過去のpathは残る。
    """
    view = load_report(report_run_id)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        save_report(view, load_surrogate_runs(view.run_ids), tuning, root)
        # 書いたものは**保存先を見て数える** (描く側に path 列を返させない = 段の
        # 綴りを 2 箇所で組まない)。
        written = sorted(
            str(file.relative_to(root)) for file in root.rglob("*") if file.is_file()
        )
        mlflow.MlflowClient().log_artifacts(report_run_id, temporary)
    logger.info("成果物 %d 件をレポート run へ保存: %s", len(written), report_run_id)
    return written
