"""MLflow I/O = **MLflow を知る唯一の場所**。3 つの experiment を持つ。

- `TARGET_EXP` (学習): surrogate の pickle + meta.json を artifact に持つ run。
- `EVAL_EXP` (波形): **1 run = 1 `EvalSeries`** = 掃引点の波形をまとめた 1 artifact。
  原系の run (`kind=original`) と置換系の run (`kind=surrogate`) がフラットに並び、
  置換系は `tags.original_hash` で自分の原系を名指す。原系は掃引の内容だけで同一性が
  決まるので、学習 run を増やしても複製されない。
- `REPORT_EXP` (レポート): **1 run = 1 回の評価の単位** = 「どの学習 run 群を どの系列で
  回したか」。持つのは `EVAL_EXP` の run_id への**参照表だけ**で、波形の実体は複製
  しない (原系は複数レポートで共有される資産)。描画はこの単位を読む。

再実行はシミュが決定的 (Euler、乱数なし) なことを使って `tags.series_hash` 一致で
スキップする。`force=True` のときだけ回し直して新しい run を積む。
"""

import hashlib
import json
import logging
import os
import tempfile
from functools import cache
from pathlib import Path
from typing import cast

import joblib
import mlflow
import mlflow.artifacts
import pandas as pd
import xarray as xr
from mlflow.entities import Run
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from tqdm import tqdm

from neurosurrogate.report import ResultSet, SeriesView, series_matrix
from neurosurrogate.sim.eval import EvalSeries, SimResult
from neurosurrogate.surrogate.bundle import META_FILE, SurrogateBundle
from neurosurrogate.surrogate.meta import SurrogateMeta

TARGET_EXP = "test_static_params"

logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    """tracking 先をリポジトリ直下の `mlflow.db` に固定する (**import 時に実行**)。

    MLflow 3 の既定 tracking URI は **cwd 相対**の `sqlite:///mlflow.db` → 設定前に
    この module の関数を呼ぶと、そのとき居たディレクトリに空 DB が生えて「run が無い」
    に見える。URI を持つのはこの module なので、呼び忘れようのない import 時に張る
    (`__file__` は resolve してから辿る = cwd にも symlink にも依存しない)。
    """
    project_root = Path(__file__).resolve().parent.parent
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    # smoke test は MLFLOW_EXPERIMENT=smoke_test で本番 experiment を汚さず隔離
    # (just clean-test が丸ごと削除)。既定は本番 experiment のまま。
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", TARGET_EXP))
    # 全 run の meta 読込で artifact DL 進捗バーが大量出力 → 抑制
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"


setup_mlflow()

SURR_ARTIFACT_DIR = "surrogate"


def log_surrogate_model(surrogate: SurrogateBundle) -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        surrogate.save(tmp_str)
        mlflow.log_artifacts(tmp_str, artifact_path=SURR_ARTIFACT_DIR)


@cache
def load_surrogate_model(run_id: str) -> SurrogateBundle:
    """run_id → surrogate。**run_id ごとに 1 回だけ** artifact を DL して unpickle
    する。同じ run が一覧走査 (get_runs_df の meta 読込) と選択後のロードで最低
    2 回、marimo のセル再実行のたびに何度も要求されるため。artifact は run に対し
    不変なので、返す bundle を使い回してよい (bundle 側も load 後は書き換えない)。
    """
    logger.debug(f"Loading surrogate from run {run_id}")
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{SURR_ARTIFACT_DIR}", dst_path=tmp_str
            )
        )
        return SurrogateBundle.load(local)


@cache
def load_surrogate_meta(run_id: str) -> SurrogateMeta:
    """run の同定情報だけを読む (meta.json のみ DL)。run 一覧は全 run 分これを呼ぶ
    ので、学習成果物の pickle まで落とさない。"""
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{SURR_ARTIFACT_DIR}/{META_FILE}", dst_path=tmp_str
            )
        )
        return SurrogateMeta.from_dict(json.loads(local.read_text()))


def load_runs(run_ids: list[str]) -> list[SurrogateBundle]:
    """run_id 列 → surrogate ロード。run 選択の唯一のロード経路
    (sweep 複数 / single 1件 共通)。表示名は meta.label (runName 非依存)。"""
    return [load_surrogate_model(rid) for rid in run_ids]


def load_bundles(run_ids: list[str]) -> dict[str, SurrogateBundle]:
    """run_id 列 → run_id→surrogate。他層は表示名でなく **run_id で** surrogate を
    引く (表示名が要る描画層は `report.run_names` で解く)。"""
    return dict(zip(run_ids, load_runs(run_ids), strict=True))


def sweep_siblings(parent_id: str) -> list[str]:
    """親 run (or 単発) の run_id → その sweep 群 = 自身 + 子 (parentRunId 一致)。
    db を直接引く (runs_df 非依存)。run_selector は代表だけ出すので引数は常に親、
    単発は子ゼロ = 1 件。"""
    children = mlflow.search_runs(
        filter_string=f"tags.`{MLFLOW_PARENT_RUN_ID}` = '{parent_id}'",
        output_format="list",
    )
    return [parent_id, *[r.info.run_id for r in children]]


def get_runs_df():
    experiment = mlflow.get_experiment_by_name(TARGET_EXP)
    if experiment is None:
        raise ValueError(
            f"Experiment '{TARGET_EXP}' が見つかりません。名前を確認してください。"
        )
    all_runs_df = cast(
        pd.DataFrame, mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    )
    if all_runs_df.empty:
        raise ValueError(f"Experiment '{TARGET_EXP}' にrunが存在しません。")
    runs_df = all_runs_df.copy()
    runs_df = runs_df.sort_values("start_time", ascending=False)
    runs_df["start_time"] = runs_df["start_time"].dt.strftime("%m-%d %H:%M:%S")
    # 親 run id (hydra --multirun の子が持つ mlflow.parentRunId)。親/単発は欠損 (NaN)。
    # 代表判定 (parent_id.isna()) と sweep 兄弟導出 (sweep_siblings) の唯一の鍵。
    # .get は親子 run が皆無で列自体が無い実験でも None → 全 NaN 列に落とす。
    runs_df["parent_id"] = runs_df.get("tags.mlflow.parentRunId")
    runs_df = runs_df[
        ["tags.mlflow.runName", "run_id", "start_time", "parent_id"]
        + [c for c in runs_df.columns if "params" in c]
    ]
    # 各 run の同定情報を dataframe 列として付与 (mlflow params に依存せず meta.json
    # から直接読む)。`meta` 列があれば UI は置換互換を replace ドメインの判定関数で
    # 直接効かせられる (互換基準を UI 側に複製しない) → 表示列には含めず絞り込み専用。
    # `comp_type` は置換対象のコンパートメント種類 = モデルペアの左側。
    # 個別 DL バーは抑制し、読込ループ全体を 1 本の進捗バーに集約。
    runs_df["meta"] = [
        _safe_meta(rid) for rid in tqdm(runs_df["run_id"], desc="meta 読込")
    ]
    excluded = int(runs_df["meta"].isna().sum())
    if excluded:
        logger.info(f"surrogate 読込不可の {excluded} 件を選択対象外")
    runs_df = runs_df[runs_df["meta"].notna()].reset_index(drop=True)
    # 全 run が読込不可 = 保存形式の変更で experiment 丸ごと死んでいる。空の
    # dataframe を下流へ流すと UI 構築が意味不明な例外で落ちるのでここで止める。
    if runs_df.empty:
        raise ValueError(
            f"Experiment '{TARGET_EXP}' の {excluded} 件すべてが読込不可 "
            "(保存形式の変更)。再学習が要る: uv run scripts/main.py"
        )
    runs_df["comp_type"] = [m.comp_type.name for m in runs_df["meta"]]
    # 出自の preset は main.py が MLflow param として記録する (surrogate の pickle
    # には入れない)。列名の mlflow 依存はここで吸収し、未記録 run 込みで欠損許容。
    runs_df["preset"] = runs_df.get("params.preset")
    return runs_df


# --- 波形 experiment (1 run = 1 EvalSeries) ----------------------------------------

# 波形だけを集める experiment。**1 run = 1 系列** (掃引点の波形を順に並べた
# artifact 1 つ) で、中身は 2 種類が並ぶだけ:
#
#   kind=original  … 掃引の原系。surrogate に依存しないので `series_hash` で共有される
#   kind=surrogate … その掃引を 1 つの学習 run の surrogate で回したもの
#
# 親子関係は張らない。置換系は `original_hash` で自分の原系を名指しするので、
# 同じ原系を何本の置換系が参照しても run 階層は平坦なまま。点は run の中の並び順
# そのもの (点ごとの run も点ごとの識別子も無い)。**どの波形をまとめて 1 回の評価と
# みなすか**はこの層の関心でない (`REPORT_EXP` が持つ)。
# smoke test は本番の評価結果を汚さない別 experiment へ (学習側の MLFLOW_EXPERIMENT
# と同じ流儀)。
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
    return _by_series_hash(_series_hash(series, run_id))


def _by_series_hash(series_hash: str) -> Run | None:
    found = mlflow.search_runs(
        experiment_ids=[_eval_exp_id()],
        filter_string=f"tags.series_hash = '{series_hash}'",
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


def _run_series(name: str, series: EvalSeries, run_id: str | None, force: bool) -> str:
    """1 系列 → 波形 run の id。既に同じ掃引 (同じ surrogate) の run があればそれを
    返すだけ = **回さない** (シミュは決定的)。`force=True` は無条件に回し直して新しい
    run を積む。"""
    found = None if force else _find_eval(series, run_id)
    if found is not None:
        return found.info.run_id
    return _log_series(name, series, series.simulate(), run_id)


# --- レポート experiment (1 run = 1 回の評価の単位) ------------------------------

# 「どの学習 run 群を どの系列で回したか」を 1 run として残す experiment。中身は
# `EVAL_EXP` の run_id への参照表 (`REFS_FILE`) だけで、波形の実体は持たない
# (原系は複数レポートで共有される資産なので複製しない)。
#
# 同一性は選択そのもの = (学習 run 群, 系列名の集合) のハッシュ。同じ選択で回し直せば
# 同じ run の参照表が更新される (`force` で波形 run が新しくなっても、指す先が
# 差し替わるだけでレポートは増えない)。
REPORT_EXP = os.environ.get("MLFLOW_REPORT_EXPERIMENT", "eval_report")
REFS_FILE = "refs.json"  # {系列名: {original: run_id, surrs: {学習 run_id: run_id}}}


def _report_exp_id() -> str:
    exp = mlflow.get_experiment_by_name(REPORT_EXP)
    return exp.experiment_id if exp else mlflow.create_experiment(REPORT_EXP)


def _report_hash(source_run_ids: list[str], names: list[str]) -> str:
    """**選択そのもの**が鍵 (学習 run 群 × 系列名の集合)。与えた順に依らない。"""
    key = json.dumps({"runs": sorted(source_run_ids), "series": sorted(names)})
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def _find_report(report_hash: str) -> Run | None:
    found = mlflow.search_runs(
        experiment_ids=[_report_exp_id()],
        filter_string=f"tags.report_hash = '{report_hash}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    return found[0] if found else None


def _log_report(
    source_run_ids: list[str], refs: dict[str, dict], report_hash: str
) -> str:
    """参照表を 1 レポート run へ。同じ選択の run が既にあれば**参照表だけ差し替える**
    (同じ選択でレポートが量産されない)。`start_run` を使わず client 直で書くのは、
    既存 run への追記が「今の active experiment」に左右されないようにするため
    (学習側の既定 experiment を張り替えないのは `_eval_exp_id` と同じ理由)。"""
    client = mlflow.MlflowClient()
    found = _find_report(report_hash)
    run_id = (
        found.info.run_id
        if found
        else _new_report_run(client, source_run_ids, refs, report_hash)
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / REFS_FILE
        # **並び順を保つ** (sort しない): 系列の順は宣言順、run の順は選択順で、
        # どちらも凡例/行見出しの並びとして描画層まで効く。
        path.write_text(json.dumps(refs, indent=2))
        client.log_artifact(run_id, str(path))
    logger.info("レポート run 保存: %s (%s)", report_hash, run_id)
    return run_id


def _new_report_run(
    client: mlflow.MlflowClient,
    source_run_ids: list[str],
    refs: dict[str, dict],
    report_hash: str,
) -> str:
    """空のレポート run を 1 本立てる (params/tags まで。参照表は呼び出し側が書く)。
    param は MLflow UI での絞り込み用の索引で、読み戻しは `report_hash` だけを見る。"""
    run = client.create_run(
        _report_exp_id(),
        tags={
            "report_hash": report_hash,
            "mlflow.runName": f"report [{report_hash}]",
        },
    )
    rid = run.info.run_id
    client.log_param(rid, "source_run_ids", json.dumps(sorted(source_run_ids)))
    client.log_param(rid, "series_names", json.dumps(sorted(refs)))
    client.set_terminated(rid)
    return rid


def run_and_log(
    bundles: dict[str, SurrogateBundle],
    catalog: dict[str, EvalSeries],
    force: bool = False,
) -> str:
    """評価実行 + 波形 run 保存 + レポート run 保存 (marimo の評価ボタンが呼ぶ唯一の
    関数)。系列ごとに原系を 1 本と置換系を学習 run ごとに 1 本ずつ確保し (既にあれば
    再利用 = 回さない)、その run_id を参照表に畳んで 1 レポートへ。返すのはレポート
    run の id。

    run 軸を掛ける組合せは `report.series_matrix` が決める (その場で回す側の
    `ResultSet.simulate` と同じ単一源)。"""
    refs = {
        name: {
            "original": _run_series(name, original, None, force),
            "surrs": {
                run_id: _run_series(name, series, run_id, force)
                for run_id, series in surrs.items()
            },
        }
        for name, original, surrs in series_matrix(catalog, bundles)
    }
    return _log_report(list(bundles), refs, _report_hash(list(bundles), list(catalog)))


def _datasets_of(run: Run) -> list[xr.Dataset]:
    """評価 run → 点の順に並んだ波形列 (artifact を読む)。"""
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            f"runs:/{run.info.run_id}/{WAVES_FILE}", dst_path=tmp
        )
        return cast(list[xr.Dataset], joblib.load(local))


def _results_of(eval_run_id: str) -> list[SimResult]:
    """波形 run の id → 点列の結果。掃引の定義が run に載っているので、点の並べ直しも
    点ごとの識別子も要らない (`EvalSeries.attach` が貼る)。"""
    run = mlflow.get_run(eval_run_id)
    series = EvalSeries.from_dict(json.loads(run.data.params["series"]))
    return series.attach(_datasets_of(run))


def load_report(source_run_ids: list[str], names: list[str]) -> ResultSet:
    """レポート run (選択そのもので引く) → 描画層の `ResultSet`。

    参照表が「どの波形をまとめて 1 回の評価とみなすか」の単一源なので、ここは
    run_id を辿って波形を読むだけ (タグ検索も原系の逆引きも要らない)。`sources` は
    成果物の由来 (`meta.json`) 用で、結果そのもの (`SimResult`) には持たせない。"""
    report_hash = _report_hash(source_run_ids, names)
    run = _find_report(report_hash)
    if run is None:
        raise ValueError(
            f"この選択のレポート run ({report_hash}) が無い。先に評価を回すこと。"
        )
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            f"runs:/{run.info.run_id}/{REFS_FILE}", dst_path=tmp
        )
        refs: dict[str, dict] = json.loads(Path(local).read_text())
    return ResultSet(
        {
            name: SeriesView(
                name,
                _results_of(ref["original"]),
                {rid: _results_of(eid) for rid, eid in ref["surrs"].items()},
                (ref["original"], *ref["surrs"].values()),
            )
            for name, ref in refs.items()
        }
    )


def _safe_meta(run_id: str) -> SurrogateMeta | None:
    """読込不可 (旧形式など) は None にして選択対象から外す。1 件の失敗で
    experiment 全体を見られなくしない。"""
    try:
        return load_surrogate_meta(run_id)
    except Exception as e:
        logger.debug(f"run {run_id} の meta 読込失敗: {e}")
        return None
