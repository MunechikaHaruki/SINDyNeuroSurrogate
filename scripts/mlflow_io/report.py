"""レポート experiment (`REPORT_EXP`): **1 run = 1 レポート = 1 系列 × N モデル**。

中身は**波形 run (`series` module) の id 2 フィールドだけ**:

    original_series_id    … 原系の波形 run 1 本
    surrogate_series_ids  … 置換系の波形 run (与えた順)

波形の実体は持たない (原系は複数レポートで共有される資産なので複製しない)。
**カタログを参照する値は持たない** — 系列名も学習 run 群も波形 run 側が持っており
(`series.name_of` / `series.source_run_of`)、そちらは「回したときの記録」なので
カタログを書き換えても過去のレポートは読める。

同一性は選択そのもの = (学習 run 群, 掃引の**内容**ハッシュ)。系列名でなく内容で
引くので、カタログの名前を変えても同じレポートに当たる。同じ選択で回し直せば同じ
run の参照が差し替わる (`force` で波形 run が新しくなってもレポートは増えない) →
参照は param でなく**書き換えられる tag** に置く。

marimo の評価ボタンは `run_and_log`、描画前の参照解決は `load_report`、run 横断の
成果物は `report_entries` を使う。書き出しそのものは `save.save_entries`。
"""

import hashlib
import json
import os

import mlflow
from mlflow.entities import Run
from tuning import Tuning

from neurosurrogate.report import SeriesView, series_matrix
from neurosurrogate.report.report import summary_figs, wave_report_figs
from neurosurrogate.sim.eval import EvalSeries
from neurosurrogate.surrogate.bundle import SurrogateBundle

from . import logger
from .save import SaveEntry, stage
from .series import name_of, results_of, run_series, source_run_of

REPORT_EXP = os.environ.get("MLFLOW_REPORT_EXPERIMENT", "eval_report")
ORIGINAL_TAG = "original_series_id"
SURROGATE_TAG = "surrogate_series_ids"  # 波形 run id の列 (与えた順 = 凡例の並び)


def _report_exp_id() -> str:
    """書く側だけが呼ぶ (無ければ作る)。読む側は `_report_exp_id_if_any`。"""
    exp = mlflow.get_experiment_by_name(REPORT_EXP)
    return exp.experiment_id if exp else mlflow.create_experiment(REPORT_EXP)


def _report_exp_id_if_any() -> str | None:
    """探すだけの経路は experiment を作らない (選択を変えるたびに空の experiment が
    生えるのを防ぐ = 探索は読み取り専用)。"""
    exp = mlflow.get_experiment_by_name(REPORT_EXP)
    return exp.experiment_id if exp else None


def _report_hash(source_run_ids: list[str], series: EvalSeries) -> str:
    """**選択そのもの**が鍵 (学習 run 群 × 掃引 1 つ)。run の与えた順に依らない。
    掃引は**内容**で効かせる (`EvalSeries.hash`) = カタログの名前を付け替えても同じ
    レポートに当たり、名前が同じでも中身を変えれば別のレポートになる。"""
    key = json.dumps({"runs": sorted(source_run_ids), "series": series.hash()})
    return hashlib.sha1(key.encode()).hexdigest()[:8]


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
    """波形 run の id 2 つを 1 レポート run へ。同じ選択の run が既にあれば**参照先だけ
    差し替える** (同じ選択でレポートが量産されない) → param でなく tag に置く
    (param は同じ run で値を変えられない)。`start_run` を使わず client 直で書くのは、
    既存 run への追記が「今の active experiment」に左右されないようにするため
    (学習側の既定 experiment を張り替えないのは `_eval_exp_id` と同じ理由)。

    `name` は run 名 (表示) にだけ使う — レポートが持つ値ではないので、カタログの
    名前が変わっても既存レポートの読み出しには効かない。"""
    client = mlflow.MlflowClient()
    found = _find_report(report_hash)
    run_id = found.info.run_id if found else _new_report_run(client, name, report_hash)
    client.set_tag(run_id, ORIGINAL_TAG, original)
    # **与えた順を保つ** (sort しない): 選択順が凡例/行見出しの並びとして
    # 描画層まで効く。
    client.set_tag(run_id, SURROGATE_TAG, json.dumps(surrs))
    logger.info("レポート run 保存: %s %s (%s)", name, report_hash, run_id)
    return run_id


def _new_report_run(client: mlflow.MlflowClient, name: str, report_hash: str) -> str:
    """空のレポート run を 1 本立てる (同一性の tag と表示名まで。参照先は
    呼び出し側が書く)。"""
    run = client.create_run(
        _report_exp_id(),
        tags={
            "report_hash": report_hash,
            "mlflow.runName": f"{name} [{report_hash}]",
        },
    )
    client.set_terminated(run.info.run_id)
    return run.info.run_id


def find_report_run(source_run_ids: list[str], series: EvalSeries) -> str | None:
    """選択 (学習 run 群 × 掃引 1 つ) → 既存レポート run の id (無ければ None)。
    **選択からレポート run_id へ渡す唯一の橋**で、以降 (描画) はこの id 1 つだけを
    見る = 描く側は「どう回したか」を再構成しない。"""
    found = _find_report(_report_hash(source_run_ids, series))
    return found.info.run_id if found else None


def run_and_log(
    bundles: dict[str, SurrogateBundle],
    name: str,
    series: EvalSeries,
    force: bool = False,
) -> str:
    """**1 系列**の評価実行 + 波形 run 保存 + レポート run 保存 (marimo の評価ボタンが
    呼ぶ唯一の関数)。原系を 1 本と置換系を学習 run ごとに 1 本ずつ確保し (既にあれば
    再利用 = 回さない)、その波形 run の id 2 フィールドを 1 レポート run へ
    (1 レポート = 1 系列 × N モデル)。返すのはそのレポート run の id
    = **そのまま描画の入力**。

    run 軸を掛ける組合せは `report.series_matrix` が決める (その場で回す側の
    `report.simulate_views` と同じ単一源)。1 本も置換できない系列は回す意味が無い
    ので落ちる (marimo の選択肢は置換できる系列だけなので通常起きない)。"""
    matrix = series_matrix({name: series}, bundles)
    if not matrix:
        raise ValueError(f"{name}: 選択 run のどれでも置換できない (比較対象が無い)")
    _, original, surrs = matrix[0]
    return _log_report(
        name,
        run_series(name, original, None, force),
        [run_series(name, s, rid, force) for rid, s in surrs.items()],
        _report_hash(list(bundles), series),
    )


def load_report(report_run_id: str) -> SeriesView:
    """**レポート run_id 1 つ** → 描画層の `SeriesView` 1 個 (1 レポート = 1 系列)。

    描画の入力はこの id と描き方 (`Tuning`) だけ = 学習 run 群も系列名も渡さない。
    レポートが持つのは**波形 run の id 2 つだけ**で、系列名 (表示) も学習 run との
    対応も波形 run 側から解く (`series.name_of` / `series.source_run_of`) = レポートは
    カタログにも学習 experiment にも依存しない。`sources` は成果物の由来
    (`meta.json`) 用で、結果そのもの (`SimResult`) には持たせない。"""
    tags = mlflow.get_run(report_run_id).data.tags
    original = tags[ORIGINAL_TAG]
    surrs: list[str] = json.loads(tags[SURROGATE_TAG])
    return SeriesView(
        name_of(original),
        results_of(original),
        {source_run_of(eid): results_of(eid) for eid in surrs},
        original,
        {source_run_of(eid): eid for eid in surrs},
    )


def report_entries(
    view: SeriesView,
    bundles: dict[str, SurrogateBundle],
    tuning: Tuning,
    report_run_id: str,
) -> list[SaveEntry]:
    """1 レポート → **レポート run に属する成果物** (`report/<レポート run>/`) =
    run 横断でこの選択でしか出ない図。

    由来は成果物ごとに違う: サマリ表は比べた**学習 run 群**、波形格子と折れ線は
    読んだ**波形 run 群** (`view.sources`)。図の名前で振り分けないよう、描画層が
    その 2 つを別の関数として返す。
    """
    dest = stage("report", report_run_id, view.name)
    return [
        SaveEntry(dest, artifact, tuple(bundles), tuning)
        for artifact in summary_figs(bundles)
    ] + [
        SaveEntry(dest, artifact, view.sources, tuning)
        for artifact in wave_report_figs(
            view, bundles, tuning.eval_comp, tuning.metric, tuning.metric_ylim
        )
    ]
