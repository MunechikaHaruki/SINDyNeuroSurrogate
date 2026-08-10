"""MLflow I/O = **MLflow を知る唯一の場所**。2 つの experiment を持つ。

- `TARGET_EXP` (学習): surrogate の pickle + meta.json を artifact に持つ run。
- `EVAL_EXP` (評価): **1 run = 1 `EvalSeries`** = 掃引点の波形をまとめた 1 artifact。
  原系の run (`kind=original`) と置換系の run (`kind=surrogate`) がフラットに並び、
  置換系は `tags.original_hash` で自分の原系を、`tags.source_run_id` で学習 run を
  名指す。原系は掃引の内容だけで同一性が決まるので、学習 run を増やしても複製
  されない。

再実行はシミュが決定的 (Euler、乱数なし) なことを使って `tags.series_hash` 一致で
スキップする。`force=True` のときだけ回し直して新しい run を積む。
"""

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

from neurosurrogate.eval import EvalSeries, SimResult
from neurosurrogate.report import ResultSet, SeriesView, series_matrix
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


# --- 評価 experiment (1 run = 1 EvalSeries) ----------------------------------------

# 評価結果だけを集める experiment。**1 run = 1 系列** (掃引点の波形を順に並べた
# artifact 1 つ) で、中身は 2 種類が並ぶだけ:
#
#   kind=original  … 掃引の原系。surrogate に依存しないので `series_hash` で共有される
#   kind=surrogate … その掃引を 1 つの学習 run の surrogate で回したもの
#
# 親子関係は張らない。置換系は `original_hash` で自分の原系を名指しするので、
# 同じ原系を何本の置換系が参照しても run 階層は平坦なまま。点は run の中の並び順
# そのもの (点ごとの run も点ごとの識別子も無い)。
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
    source_run_id: str | None,
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
                **({"source_run_id": source_run_id} if source_run_id else {}),
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


def run_and_log(
    bundles: dict[str, SurrogateBundle],
    catalog: dict[str, EvalSeries],
    source_run_id: str,
    force: bool = False,
) -> list[str]:
    """評価実行 + 評価 run 保存 (marimo の評価ボタンが呼ぶ唯一の関数)。系列ごとに
    原系を 1 本 (既にあれば再利用 = **原系の遅延実行**) と、置換系を run ごとに 1 本
    ずつ積む。既に同じ掃引の run があれば系列ごとスキップする (`force=True` で
    回し直す)。返すのは保存した置換系 run の id。

    run 軸を掛ける組合せは `report.series_matrix` が決める (その場で回す側の
    `ResultSet.simulate` と同じ単一源)。"""
    logged: list[str] = []
    for name, original, surrs in series_matrix(catalog, bundles):
        pending = {
            run_id: series
            for run_id, series in surrs.items()
            if force or not _find_eval(series, run_id)
        }
        if not pending:
            continue
        # 原系は置換系が 1 本でも要るときにだけ回す (無ければ保存もしない)。
        if force or not _find_eval(original, None):
            _log_series(name, original, original.simulate(), None, None)
        for run_id, series in pending.items():
            logged.append(
                _log_series(name, series, series.simulate(), run_id, source_run_id)
            )
    return logged


def _datasets_of(run: Run) -> list[xr.Dataset]:
    """評価 run → 点の順に並んだ波形列 (artifact を読む)。"""
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            f"runs:/{run.info.run_id}/{WAVES_FILE}", dst_path=tmp
        )
        return cast(list[xr.Dataset], joblib.load(local))


def _results_of(run: Run) -> tuple[str, EvalSeries, list[SimResult]]:
    """評価 run → (系列名, 掃引, 点列の結果)。掃引の定義が run に載っているので、
    点の並べ直しも点ごとの識別子も要らない (`EvalSeries.attach` が貼る)。"""
    series = EvalSeries.from_dict(json.loads(run.data.params["series"]))
    return run.data.params["name"], series, series.attach(_datasets_of(run))


def load_eval_results(source_run_ids: list[str]) -> ResultSet:
    """学習 run 群 (sweep 兄弟) が出した評価結果 → 描画層の `ResultSet`。

    置換系の run を引き、それぞれが名指しする原系 (`original_hash`) を辿るだけ =
    原系が 1 本しか無くても run 軸の全系列に同じ原系が付く。系列名が同じ run は
    同じ `SeriesView` の列になる。`sources` は成果物の由来 (`meta.json`) 用で、
    結果そのもの (`SimResult`) には持たせない。"""
    points: dict[str, list[SimResult]] = {}
    surrs: dict[str, dict[str, list[SimResult]]] = {}
    sources: dict[str, dict[str, None]] = {}
    for source_run_id in source_run_ids:
        for run in mlflow.search_runs(
            experiment_ids=[_eval_exp_id()],
            filter_string=f"tags.source_run_id = '{source_run_id}'",
            output_format="list",
        ):
            name, _series, results = _results_of(run)
            surrs.setdefault(name, {})[run.data.params["run_id"]] = results
            sources.setdefault(name, {})[run.info.run_id] = None
            if name not in points:
                original = _by_series_hash(run.data.tags["original_hash"])
                if original is None:
                    raise ValueError(f"{name}: 原系の評価 run が見つからない")
                points[name] = _results_of(original)[2]
                sources[name][original.info.run_id] = None
    return ResultSet(
        {
            name: SeriesView(name, points[name], surrs[name], tuple(sources[name]))
            for name in surrs
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
