"""レポート experiment (`REPORT_EXP`): **1 run = 1 レポート = 1 系列 × N モデル**。

持つのは波形 run の id 2 tag だけで、波形もカタログ由来の値も持たない。同一性は選択
そのもの (`sim.spec.EvalSelection.hash`) で、同じ選択で回し直すと参照だけ差し替わる
→ param でなく tag。marimo の評価ボタンが `run_report`。

**この module は書く/読むだけ** — 読んだ中身をどの図のどの段に並べるかは
`artifact.bundle.build_report` (組立に `Tuning` と surrogate の中身が要り、
MLflow の関心でない)。
"""

import json
import os

import mlflow
from catalog import SERIES

from neurosurrogate.sim.result import SeriesResults
from neurosurrogate.sim.run import replaced_runs
from neurosurrogate.sim.spec import EvalSelection
from neurosurrogate.surrogate.bundle import SurrogateBundle

from . import logger
from ._query import exp_id, latest_by_tag
from .series import load_column, run_series

REPORT_EXP = os.environ.get("MLFLOW_REPORT_EXPERIMENT", "eval_report")
_HASH_TAG = "report_hash"  # 同一性 = 選択そのもの (EvalSelection.hash)
_ORIGINAL_TAG = "original_series_id"
_SURROGATE_TAG = "surrogate_series_ids"  # 波形 run id の列 (与えた順 = 凡例の並び)


def _report_hash(name: str, run_ids: tuple[str, ...]) -> str:
    """同一性の鍵 = 選択そのもの (系列名 × 学習 run 群)。**鍵を組むのはここだけ** =
    書く側と探す側が同じ綴りに揃い、カタログ (`SERIES`) を引くのも 1 箇所に閉じる。"""
    return EvalSelection(SERIES[name], run_ids).hash()


def _log_report(
    name: str, original: str, surrs: list[str], run_ids: tuple[str, ...]
) -> str:
    """波形 run の id 2 つを 1 レポート run へ。同じ選択の run があれば**参照先だけ
    差し替える** → param でなく tag。`name` は鍵と表示に使う。"""
    # client 直で書く: 既存 run への追記を active experiment に左右されないため。
    client = mlflow.MlflowClient()
    report_hash = _report_hash(name, run_ids)
    found = latest_by_tag(REPORT_EXP, _HASH_TAG, report_hash)
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
        exp_id(REPORT_EXP),
        tags={
            _HASH_TAG: report_hash,
            "mlflow.runName": f"{name} [{report_hash}]",
        },
    )
    client.set_terminated(run.info.run_id)
    return run.info.run_id


def find_report_run(name: str, run_ids: tuple[str, ...]) -> str | None:
    """選択 (系列名 × 学習 run 群) → 既存レポート run の id (無ければ None)。**選択から
    レポート run_id へ渡す唯一の橋**で、以降 (描画) はこの id 1 つだけを見る。"""
    found = latest_by_tag(REPORT_EXP, _HASH_TAG, _report_hash(name, run_ids))
    return found.info.run_id if found else None


def run_report(bundles: dict[str, SurrogateBundle], name: str) -> str:
    """1 系列の評価実行 + 波形 run 保存 + レポート run 保存 (marimo の評価ボタン)。
    系列は**名前から引く** = 呼ぶ側はカタログを触らない。既にある波形 run は再利用
    = 回さない。返りはレポート run の id = そのまま描画の入力。1 本も置換できない選択は
    回す意味が無いので拒む。"""
    series = SERIES[name]
    surrs = replaced_runs(series, bundles)
    if not surrs:
        raise ValueError(f"{name}: 選択 run のどれでも置換できない (比較対象が無い)")
    return _log_report(
        name,
        run_series(name, series, None, None),
        [run_series(name, series, rid, bundle) for rid, bundle in surrs.items()],
        tuple(bundles),
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
