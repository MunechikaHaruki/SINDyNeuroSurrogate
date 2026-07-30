"""MLflow I/O = **MLflow を知る唯一の場所**。2 つの experiment を持つ。

- `TARGET_EXP` (学習): surrogate の pickle + meta.json を artifact に持つ run。
- `EVAL_EXP` (評価): 1 run = 1 `SimSpec` = 1 波形。**親 run = 原系 / 子 run =
  置換系**で、同じ刺激条件の原系と置換系が 1 グループになる (比較の単位が run 階層
  そのもの)。原系は学習 run に依らないので刺激条件ごとに 1 本だけ作られ、学習 run
  を増やしても複製されない。子は `tags.source_run_id` で学習 run を指す。

再実行はシミュが決定的 (Euler、乱数なし) なことを使って `tags.spec_hash` 一致で
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

from neurosurrogate.eval import SimSpec, simulate
from neurosurrogate.runs import SimKey, SimResult, expand, run_labels
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


def load_bundles(
    run_ids: list[str],
) -> tuple[dict[str, SurrogateBundle], dict[str, str]]:
    """run_id 列 → (run_id→surrogate, run_id→表示名)。`neurosurrogate.eval` は
    surrogate を **run_id をキーに**扱う (`SimSpec.run_id` が MLflow run_id
    そのものなので、他層は表示名でなく run_id で引く) → ここで揃えて返す。
    marimo からはこれ 1 回の呼び出しで済ませる。"""
    bundles = load_runs(run_ids)
    names = dict(zip(run_ids, run_labels(bundles), strict=True))
    return dict(zip(run_ids, bundles, strict=True)), names


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


# --- 評価 experiment (1 run = 1 SimSpec) ------------------------------------------

# smoke test は本番の評価結果を汚さない別 experiment へ (学習側の MLFLOW_EXPERIMENT
# と同じ流儀)。
EVAL_EXP = os.environ.get("MLFLOW_EVAL_EXPERIMENT", "eval")
WAVE_FILE = "wave.joblib"  # 波形 artifact のファイル名 (1 run に 1 つ)
WAVE_DTYPE = "float32"  # 保存精度 (表示にも指標にも十分で容量は半分)
_KIND_ORIGINAL = "original"
_KIND_SURROGATE = "surrogate"


def _eval_exp_id() -> str:
    """評価 experiment の id (無ければ作る)。`set_experiment` は学習側の既定を
    書き換えてしまうので使わず、run ごとに experiment_id を指定する。"""
    exp = mlflow.get_experiment_by_name(EVAL_EXP)
    return exp.experiment_id if exp else mlflow.create_experiment(EVAL_EXP)


def _find_eval(spec: SimSpec) -> Run | None:
    """同じ入力の評価 run (最新)。決定的なシミュなので、あれば回し直す必要はない。"""
    found = mlflow.search_runs(
        experiment_ids=[_eval_exp_id()],
        filter_string=f"tags.spec_hash = '{spec.hash()}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    return found[0] if found else None


def _log_eval(
    label: str, result: SimResult, source_run_id: str | None, parent_id: str | None
) -> str:
    """1 SimResult を 1 評価 run へ。`spec` param が入力仕様の単一源 (読み戻しは
    ここから)、平坦化した param は MLflow UI での絞り込み/比較用の索引。"""
    spec = result.spec
    with mlflow.start_run(experiment_id=_eval_exp_id(), run_name=label) as run:
        mlflow.log_params(
            {
                "spec": spec.key(),
                "label": label,
                "name": spec.name,
                "target": spec.target,
                "current_type": spec.current_type,
                "dt": spec.dt,
                "sweep_param": spec.sweep_param,
                "sweep_value": spec.sweep_value,
                **{f"cp.{k}": v for k, v in spec.current_params.items()},
            }
        )
        mlflow.set_tags(
            {
                "spec_hash": spec.hash(),
                "kind": _KIND_ORIGINAL if parent_id is None else _KIND_SURROGATE,
                **({"run_label": result.run_label} if result.run_label else {}),
                **({"source_run_id": source_run_id} if source_run_id else {}),
                **({MLFLOW_PARENT_RUN_ID: parent_id} if parent_id else {}),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / WAVE_FILE
            joblib.dump(
                result.dataset.map(lambda v: v.astype(WAVE_DTYPE), keep_attrs=True),
                path,
                compress=1,  # float32 波形はほぼ非圧縮 → 高レベルは時間の無駄
            )
            mlflow.log_artifact(str(path))
        logger.info("評価 run 保存: %s (%s)", label, run.info.run_id)
        return run.info.run_id


def run_and_log(
    bundles: dict[str, SurrogateBundle],
    specs: dict[str, SimSpec],
    run_labels_by_id: dict[str, str],
    source_run_id: str,
    force: bool = False,
) -> list[str]:
    """評価実行 + 評価 run 保存 (marimo の評価ボタンが呼ぶ唯一の関数)。原系を親、
    置換系をその子として積む。既に同じ入力の run があればシミュごとスキップする
    (`force=True` で回し直す)。返すのは保存した子 run の id。"""
    if not bundles:
        return []
    expanded = expand(specs, bundles)
    parents: dict[str, str] = {}  # label → 原系 run_id
    logged: list[str] = []
    for (label, run_id), spec in expanded.items():
        found = None if force else _find_eval(spec)
        if run_id is None:
            parents[label] = (
                found.info.run_id
                if found
                else _log_eval(label, _simulated(spec, None), None, None)
            )
            continue
        if found:
            continue
        result = _simulated(spec, bundles[run_id], run_labels_by_id[run_id])
        logged.append(_log_eval(label, result, source_run_id, parents[label]))
    return logged


def _simulated(
    spec: SimSpec, bundle: SurrogateBundle | None, run_label: str | None = None
) -> SimResult:
    return SimResult(spec, run_label, simulate(spec, bundle))


def _result_of(run: Run) -> SimResult:
    """評価 run → SimResult (波形 artifact を読む)。"""
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            f"runs:/{run.info.run_id}/{WAVE_FILE}", dst_path=tmp
        )
        dataset = cast(xr.Dataset, joblib.load(local))
    return SimResult(
        SimSpec.from_dict(json.loads(run.data.params["spec"])),
        run.data.tags.get("run_label"),
        dataset,
        source=run.info.run_id,
    )


def load_eval_results(source_run_ids: list[str]) -> dict[SimKey, SimResult]:
    """学習 run 群 (sweep 兄弟) が出した評価結果 → `SimKey → SimResult`。子 run
    (置換系) を引き、その親 run から原系を辿る = 原系が 1 本しか無くても run 軸の
    全系列に対して同じ原系が付く。"""
    out: dict[SimKey, SimResult] = {}
    for source_run_id in source_run_ids:
        for child in mlflow.search_runs(
            experiment_ids=[_eval_exp_id()],
            filter_string=f"tags.source_run_id = '{source_run_id}'",
            output_format="list",
        ):
            label = child.data.params["label"]
            result = _result_of(child)
            out[(label, result.spec.run_id)] = result
            if (label, None) not in out:
                parent = mlflow.get_run(child.data.tags[MLFLOW_PARENT_RUN_ID])
                out[(label, None)] = _result_of(parent)
    return out


def _safe_meta(run_id: str) -> SurrogateMeta | None:
    """読込不可 (旧形式など) は None にして選択対象から外す。1 件の失敗で
    experiment 全体を見られなくしない。"""
    try:
        return load_surrogate_meta(run_id)
    except Exception as e:
        logger.debug(f"run {run_id} の meta 読込失敗: {e}")
        return None
