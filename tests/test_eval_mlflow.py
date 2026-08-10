"""評価結果の永続化 (MLflow の評価 experiment) の smoke。

`tests/test_surrogate.py` がドメイン層だけを通すのに対し、ここは
`scripts/mlflow_io.py` = **MLflow を知る唯一の場所**を通す。tracking 先は tmp の
sqlite へ丸ごと差し替えるので、手元の `mlflow.db` / `mlruns/` は汚れない。
"""

import sys
from pathlib import Path

import mlflow
import numpy as np
import pytest
from test_surrogate import fit_surrogate

from neurosurrogate.core import access
from neurosurrogate.eval import EvalSeries, SimSpec, simulate
from neurosurrogate.surrogate.bundle import SurrogateBundle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mlflow_io  # noqa: E402

RUN_ID = "RID"  # 学習 run の代役 (評価 run が指す先)


@pytest.fixture
def eval_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """tracking 先を tmp へ移し、評価 experiment 名もテスト専用にする。"""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(mlflow_io, "EVAL_EXP", "test_eval")
    return "test_eval"


@pytest.fixture(scope="module")
def sindy() -> SurrogateBundle:
    return fit_surrogate("_test_hh_sindy")


def _evals(bundle: SurrogateBundle) -> dict[str, EvalSeries]:
    """学習と同じ入力を**さらに短く**した評価仕様 (2 点の掃引)。ここで見たいのは
    保存/読込の往復であって波形の質ではないので、シミュ長は最小で足りる。掃引に
    してあるのは、点の並びが `EvalSeries.points` から復元されることを見るため。"""
    ds = bundle.meta.dataset
    return {
        "hh_dc": EvalSeries(
            spec=SimSpec(
                target=ds.model_name,
                current_type=ds.current_type,
                dt=ds.dt,
                current_params=dict(ds.current_params),
            ),
            param="duration",
            values=[170.0, 190.0],
        )
    }


def test_eval_runs_round_trip_without_resimulating(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """**1 run = 1 EvalSeries**。掃引点の波形をまとめた 1 artifact が往復し、再シミュ
    無しに点の並びごと戻る。同じ掃引の再実行はスキップされ、原系の run は学習 run を
    跨いで共有される (親子関係は使わず `original_hash` で辿る)。"""
    evals = _evals(sindy)
    series = evals["hh_dc"]
    logged = mlflow_io.run_and_log({RUN_ID: sindy}, evals, RUN_ID)
    assert len(logged) == 1  # 置換系 1 本 (点ごとには分かれない)

    view = mlflow_io.load_eval_results([RUN_ID])["hh_dc"]
    assert view.run_ids == [RUN_ID]
    # 点は宣言した掃引値の順で戻る (点ごとの識別子を保存していない)
    assert (view.axis, view.values) == ("duration", [170.0, 190.0])
    orig, surr = view.pair(1, RUN_ID)
    # 入力仕様が往復し、そこから dataset を復元できる
    assert surr.spec.dataset().model_name == series.spec.target
    # 由来 (どの評価 run から読んだか) は結果でなく `SeriesView` 側が持つ
    # (置換系 1 本 + 原系 1 本)
    assert len(view.sources) == 2
    # 波形は float32 で往復し、原系/置換系はそれぞれの run に入っている
    # (原系は surrogate 非依存なので回し直しても一致する)
    np.testing.assert_allclose(
        access.potential(orig.dataset, 0),
        access.potential(simulate(series.points[1], None).dataset, 0),
        rtol=1e-5,
    )
    assert not np.allclose(
        access.potential(surr.dataset, 0), access.potential(orig.dataset, 0)
    )

    # シミュは決定的 → 同じ掃引の 2 度目はスキップ (series_hash 一致)
    assert mlflow_io.run_and_log({RUN_ID: sindy}, evals, RUN_ID) == []

    # 別の学習 run から同じ条件 → 置換系は増えるが原系は共有される
    # (学習 run を増やしても原系の波形が複製されない)
    mlflow_io.run_and_log({"OTHER": sindy}, evals, "OTHER")
    originals = mlflow.search_runs(
        experiment_ids=[mlflow_io._eval_exp_id()],
        filter_string="tags.kind = 'original'",
        output_format="list",
    )
    assert len(originals) == 1
    both = mlflow_io.load_eval_results([RUN_ID, "OTHER"])["hh_dc"]
    assert (len(both.points), both.run_ids) == (2, [RUN_ID, "OTHER"])
