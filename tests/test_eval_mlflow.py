"""評価結果の永続化 (MLflow の評価 experiment) の smoke。

`tests/test_surrogate.py` がドメイン層だけを通すのに対し、ここは
`scripts/mlflow_io.py` = **MLflow を知る唯一の場所**を通す。tracking 先は tmp の
sqlite へ丸ごと差し替えるので、手元の `mlflow.db` / `mlruns/` は汚れない。
"""

from pathlib import Path
from typing import cast

import mlflow

# `scripts/` は conftest が import path へ入れている。
import mlflow_io  # noqa: E402
import numpy as np
import pytest
from mlflow.entities import Run
from test_surrogate import fit_surrogate

from neurosurrogate.core import access
from neurosurrogate.sim.eval import EvalSeries, simulate
from neurosurrogate.surrogate.bundle import SurrogateBundle

RUN_ID = "RID"  # 学習 run の代役 (評価 run が指す先)


@pytest.fixture
def eval_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """tracking 先を tmp へ移し、評価/レポート experiment 名もテスト専用にする。"""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(mlflow_io, "EVAL_EXP", "test_eval")
    monkeypatch.setattr(mlflow_io, "REPORT_EXP", "test_report")
    return "test_eval"


@pytest.fixture(scope="module")
def sindy() -> SurrogateBundle:
    return fit_surrogate("_test_hh_sindy")


def _evals(bundle: SurrogateBundle) -> dict[str, EvalSeries]:
    """学習と同じ入力を**さらに短く**した評価仕様 (2 点の掃引)。ここで見たいのは
    保存/読込の往復であって波形の質ではないので、シミュ長は最小で足りる。掃引に
    してあるのは、点の並びが `EvalSeries.points` から復元されることを見るため。"""
    return {
        "hh_dc": EvalSeries(
            spec=bundle.meta.dataset, param="duration", values=[170.0, 190.0]
        )
    }


def test_eval_runs_round_trip_without_resimulating(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """**1 波形 run = 1 EvalSeries、1 レポート run = 1 回の評価**。掃引点の波形を
    まとめた 1 artifact が往復し、再シミュ無しに点の並びごと戻る。同じ掃引の再実行は
    スキップされ、原系の run は学習 run を跨いで共有される。レポートは
    **1 系列 × N モデル**が単位で、選択 (学習 run 群 × 系列 1 つ) で引き、波形 run
    への参照だけを持つ。"""
    evals = _evals(sindy)
    series = evals["hh_dc"]
    report_ids = mlflow_io.run_and_log({RUN_ID: sindy}, evals)
    # 波形は 2 本 (原系 1 + 置換系 1)、レポートは系列ごとに 1 本 (点では分かれない)
    assert len(_of_kind("original")) == 1 and len(_of_kind("surrogate")) == 1
    assert list(report_ids) == ["hh_dc"]

    view = mlflow_io.load_report([RUN_ID], "hh_dc")
    assert view.run_ids == [RUN_ID]
    # 点は宣言した掃引値の順で戻る (点ごとの識別子を保存していない)
    assert (view.axis, view.values) == ("duration", [170.0, 190.0])
    orig, surr = view.pair(1, RUN_ID)
    # 入力仕様が往復し、そこから実行入力を復元できる
    assert surr.spec.materialize().net is series.spec.net
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

    # シミュは決定的 → 同じ掃引の 2 度目はスキップ (series_hash 一致)。同じ選択の
    # レポートも量産されず、同じ run が更新される。
    assert mlflow_io.run_and_log({RUN_ID: sindy}, evals) == report_ids
    assert len(_of_kind("surrogate")) == 1

    # 別の学習 run から同じ条件 → 置換系は増えるが原系は共有される
    # (学習 run を増やしても原系の波形が複製されない)。選択が違えば別のレポート。
    assert mlflow_io.run_and_log({"OTHER": sindy}, evals) != report_ids
    assert (len(_of_kind("original")), len(_of_kind("surrogate"))) == (1, 2)

    # 学習 run 2 件の選択はさらに別のレポート = 別の単位 (run 軸 2 本が 1 枚に並ぶ)
    mlflow_io.run_and_log({RUN_ID: sindy, "OTHER": sindy}, evals)
    both = mlflow_io.load_report([RUN_ID, "OTHER"], "hh_dc")
    assert (len(both.points), both.run_ids) == (2, [RUN_ID, "OTHER"])


def _of_kind(kind: str) -> list[Run]:
    """波形 experiment の run を kind で数える (原系が複製されないことの確認用)。"""
    return cast(
        list[Run],
        mlflow.search_runs(
            experiment_ids=[mlflow_io._eval_exp_id()],
            filter_string=f"tags.kind = '{kind}'",
            output_format="list",
        ),
    )
