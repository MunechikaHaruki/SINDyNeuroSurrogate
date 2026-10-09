"""評価結果の永続化 (MLflow の波形 experiment) の smoke。

`tests/test_surrogate.py` がドメイン層だけを通すのに対し、ここは
`scripts/mlflow_io/` (experiment と run、波形の保存形式) を通す。tracking 先は tmp の
sqlite へ丸ごと差し替えるので、手元の `mlflow.db` / `mlruns/` は汚れない。
"""

import json
from pathlib import Path
from typing import cast

import mlflow
import mlflow.artifacts

# `scripts/` は conftest が import path へ入れている。
import mlflow_io._series as series_io  # noqa: E402
import mlflow_io.evaluation as evaluation  # noqa: E402
import numpy as np
import pytest
from mlflow.entities import Run
from mlflow_io.surrogate import log_surrogate_model
from test_surrogate import fit_preset

from neurosurrogate.core import access
from neurosurrogate.sim.run import _simulate
from neurosurrogate.sim.spec import EvalSeries
from neurosurrogate.surrogate.model import Surrogate


def _exp_id(name: str) -> str:
    """experiment 名 → id。本番の解決子 (`mlflow_io._query`) はパッケージ内専用なので、
    テストは MLflow を直に引く。"""
    exp = mlflow.get_experiment_by_name(name)
    assert exp is not None
    return str(exp.experiment_id)


def _train_run(name: str, bundle: Surrogate) -> str:
    """名前と surrogate artifact を持つ学習 run を 1 本立てる。"""
    exp = mlflow.get_experiment_by_name("test_train")
    experiment_id = exp.experiment_id if exp else mlflow.create_experiment("test_train")
    with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:
        log_surrogate_model(bundle)
        return str(run.info.run_id)


@pytest.fixture(scope="module")
def sindy() -> Surrogate:
    return fit_preset("_test_hh_sindy")


@pytest.fixture
def eval_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sindy: Surrogate
) -> None:
    """tracking 先を tmp へ移し、波形 experiment 名もテスト専用にする。カタログも
    差し替える — `evaluate` は系列を**名前から引く**ので、テスト用の短い掃引
    (`_evals`) をその名前に載せて渡す。"""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(series_io, "_EVAL_EXP", "test_eval")
    monkeypatch.setattr(evaluation, "SERIES", {"hh_dc": _evals(sindy)})


def _evals(bundle: Surrogate) -> EvalSeries:
    """学習と同じ入力を**さらに短く**した評価仕様 (2 点の掃引)。見たいのは保存と
    読込の往復であって波形の質ではないので、シミュ長は最小で足りる。掃引にして
    あるのは、点の並びが `EvalSeries.points` から復元されることを見るため。"""
    return EvalSeries(
        spec=bundle.spec.dataset,
        replace_targets=("soma",),
        param="duration",
        values=[170.0, 190.0],
    )


def _of_kind(kind: str) -> list[Run]:
    """波形 experiment の run を kind で数える (原系が複製されないことの確認用)。"""
    return cast(
        list[Run],
        mlflow.search_runs(
            experiment_ids=[_exp_id(series_io._EVAL_EXP)],
            filter_string=f"tags.kind = '{kind}'",
            output_format="list",
        ),
    )


def _artifact(run: Run, path: str) -> Path:
    return Path(mlflow.artifacts.download_artifacts(f"runs:/{run.info.run_id}/{path}"))


def test_eval_runs_round_trip_without_resimulating(
    eval_store: None, sindy: Surrogate
) -> None:
    """**1 波形 run = 1 列**。点の並びごと波形が往復し、同じ掃引の再実行は回さず、
    原系の run は学習 run を跨いで共有される。置換系は `original_hash` で原系を指す。"""
    series = _evals(sindy)
    train_id = _train_run("RID", sindy)
    evaluation.evaluate("hh_dc", (train_id,))
    (original,), (surrogate,) = _of_kind("original"), _of_kind("surrogate")
    assert surrogate.data.tags["original_hash"] == original.data.tags["series_hash"]
    assert surrogate.data.tags["series_name"] == "hh_dc"

    column = series_io._load_column(original.info.run_id)
    # 記述が往復し、点は宣言した掃引値の順で戻る
    assert column.series.replaced_hash() == series.replaced_hash()
    assert (column.series.param, column.series.axis_values) == (
        "duration",
        [170.0, 190.0],
    )
    # 波形は float32 で往復する (原系は surrogate 非依存なので回し直しても一致する)
    np.testing.assert_allclose(
        access.potential(column.waves[1], 0),
        access.potential(_simulate(series.points[1], None, ()), 0),
        rtol=1e-5,
    )
    surr = series_io._load_column(surrogate.info.run_id)
    assert surr.run_id == train_id
    assert not np.allclose(
        access.potential(surr.waves[1], 0), access.potential(column.waves[1], 0)
    )

    # シミュは決定的 → 同じ評価の 2 度目は回さない (series_hash 一致)
    evaluation.evaluate("hh_dc", (train_id,))
    assert (len(_of_kind("original")), len(_of_kind("surrogate"))) == (1, 1)
    # 別の学習 run から同じ条件 → 置換系は増えるが原系は共有される
    evaluation.evaluate("hh_dc", (train_id, _train_run("OTHER", sindy)))
    assert (len(_of_kind("original")), len(_of_kind("surrogate"))) == (1, 2)


def test_surrogate_run_holds_what_the_figures_read(
    eval_store: None, sindy: Surrogate
) -> None:
    """冊が読むものは全部置換系の run の中にある: 素の float32 の波形、全 comp・全点の
    指標、置換した comp の射影、学習側の指標。JSON に nan は出さない (JS が読めない)。
    """
    evaluation.evaluate("hh_dc", (_train_run("RID", sindy),))
    (original,), (surrogate,) = _of_kind("original"), _of_kind("surrogate")

    meta = json.loads(_artifact(original, "waves/meta.json").read_text())
    assert meta["names"] == {"soma": 0}
    n_vars = sum(1 for comp, _, _ in meta["features"] if comp == 0)
    raw = np.fromfile(_artifact(original, "waves/1/comp0.f32"), dtype="<f4")
    assert raw.size == n_vars * meta["n"][1]
    potential = access.potential(
        series_io._load_column(original.info.run_id).waves[1], 0
    )
    np.testing.assert_array_equal(raw[: meta["n"][1]], potential)

    def strict(path: str) -> dict:
        def reject(token: str) -> None:
            raise ValueError(token)

        text = _artifact(surrogate, path).read_text()
        return cast(dict, json.loads(text, parse_constant=reject))

    metrics = strict("metrics.json")
    assert list(metrics) == ["soma"] and len(metrics["soma"]) == 2
    assert "nnz" in strict("summary.json")
    projected = json.loads(_artifact(surrogate, "projected/soma/meta.json").read_text())
    assert [v for _, v, _ in projected["features"]][0] == "V"


def test_evaluate_rejects_series_no_run_can_replace(eval_store: None) -> None:
    """1 本も置換できない選択は回す意味が無い → 原系だけ回さずに落ちる。"""
    with pytest.raises(ValueError, match="置換できない"):
        evaluation.evaluate("hh_dc", ())
