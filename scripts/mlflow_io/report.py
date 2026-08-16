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

marimo の評価ボタンは `run_and_log`、描画前の参照解決は `load_report`、描く成果物は
`report_artifacts` / `series_artifacts`。書き出しは `save.save_artifacts` が
**このレポート run 自身へ** (描いたものは全部 1 レポートに束ねる)。
"""

import hashlib
import json
import os
from dataclasses import dataclass

import mlflow
from mlflow.entities import Run
from tuning import Tuning

from neurosurrogate.plotting import Artifact
from neurosurrogate.sim.figures import (
    detail_figs,
    original_figs,
    run_names,
    summary_figs,
    wave_report_figs,
)
from neurosurrogate.sim.result import SeriesResults
from neurosurrogate.sim.run import replaced_runs
from neurosurrogate.sim.spec import EvalSeries
from neurosurrogate.surrogate.bundle import SurrogateBundle

from . import logger
from .save import slug, under
from .series import results_of, run_series, source_run_of

REPORT_EXP = os.environ.get("MLFLOW_REPORT_EXPERIMENT", "eval_report")
_ORIGINAL_TAG = "original_series_id"
_SURROGATE_TAG = "surrogate_series_ids"  # 波形 run id の列 (与えた順 = 凡例の並び)


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
    client.set_tag(run_id, _ORIGINAL_TAG, original)
    # **与えた順を保つ** (sort しない): 選択順が凡例/行見出しの並びとして
    # 描画層まで効く。
    client.set_tag(run_id, _SURROGATE_TAG, json.dumps(surrs))
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

    run 軸を絞るのは `sim.run.replaced_runs` (描画側と同じ単一源)。1 本も置換
    できない系列は回す意味が無いので拒む (marimo の選択肢は置換できる系列だけなので
    通常起きない)。"""
    surrs = replaced_runs(series, bundles)
    if not surrs:
        raise ValueError(f"{name}: 選択 run のどれでも置換できない (比較対象が無い)")
    return _log_report(
        name,
        run_series(name, series, None, force),
        [run_series(name, series, model, force) for model in surrs.items()],
        _report_hash(list(bundles), series),
    )


@dataclass(frozen=True)
class Report:
    """**レポート run を解いたもの**: 描く中身 (`view`) と、それをどの run から読んだか
    (`run_id` / `original_id` / `surr_ids`)。

    **id を持つのはこの層だけ** — MLflow の同一性はドメイン層 (`SeriesResults`) に
    入れず、保存段と由来 (`meta.json`) を解くここが持つ。描画関数はどれも `view` しか
    見ない。
    """

    run_id: str  # レポート run (run 横断の成果物の宛先)
    view: SeriesResults
    original_id: str  # 原系を読んだ波形 run
    surr_ids: dict[str, str]  # 学習 run → その置換系を読んだ波形 run

    def __post_init__(self) -> None:
        # id は **run 軸と揃う**のが条件。半端に欠けた id は保存段が別の run の名前へ
        # 落ちる = 由来の欠けた成果物になるので、図を描く前に構築時点で弾く。
        if (
            set(self.surr_ids) != set(self.view.surrs)
            or not all(self.surr_ids.values())
            or not self.original_id
        ):
            raise ValueError("評価 run の id が run 軸と揃わない")

    @property
    def sources(self) -> tuple[str, ...]:
        """読んだ波形 run の id (原系が先、置換系は run 軸の順)。成果物の由来
        (`meta.json`) にそのまま落ちる。"""
        return (self.original_id, *self.surr_ids.values())


def load_report(report_run_id: str) -> Report:
    """**レポート run_id 1 つ** → 描く中身と由来 (`Report`) (1 レポート = 1 系列)。

    描画の入力はこの id と描き方 (`Tuning`) だけ = 学習 run 群も系列名も渡さない。
    レポートが持つのは**波形 run の id 2 つだけ**で、系列名 (表示) も学習 run との
    対応も波形 run 側から解く (`series.name_of` / `series.source_run_of`) = レポートは
    カタログにも学習 experiment にも依存しない。"""
    tags = mlflow.get_run(report_run_id).data.tags
    original = tags[_ORIGINAL_TAG]
    surrs: list[str] = json.loads(tags[_SURROGATE_TAG])
    return Report(
        report_run_id,
        SeriesResults(
            results_of(original),
            {source_run_of(eid): results_of(eid) for eid in surrs},
        ),
        original,
        {source_run_of(eid): eid for eid in surrs},
    )


def report_artifacts(
    report: Report,
    bundles: dict[str, SurrogateBundle],
    tuning: Tuning,
) -> list[Artifact]:
    """1 レポート → **run 横断でこの選択でしか出ない図** (レポート run の直下)。
    比べた N 本のサマリ表と、点軸 × run 軸に開いた波形格子/折れ線。"""
    return [
        *summary_figs(bundles),
        *wave_report_figs(
            report.view, bundles, tuning.eval_comp, tuning.metric, tuning.metric_ylim
        ),
    ]


def series_artifacts(
    report: Report, bundles: dict[str, SurrogateBundle], tuning: Tuning
) -> list[Artifact]:
    """1 レポートの波形群 → **波形 1 本ずつで決まる図** (`series/<表示名>/`)。
    原系の入力電流と、置換系ごとの詳細図。

    段の名前が波形 run の id でなく表示名なのは、宛先がレポート run 1 本で衝突
    しないから (凡例と同じ読み方で段を引ける)。
    """
    view = report.view
    labels = run_names(bundles)
    out = under("series/original", original_figs(view))
    for run_id in view.run_ids:
        out += under(
            f"series/{slug(labels[run_id])}",
            detail_figs(
                view,
                run_id,
                bundles[run_id],
                tuning.eval_comp,
                tuning.view_comps,
                tuning.detail_point,
                tuning.spike_orig,
                tuning.spike_surr,
            ),
        )
    return out
