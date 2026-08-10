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
from neurosurrogate.eval import SimSpec, simulate
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


def _evals(bundle: SurrogateBundle) -> dict[str, dict]:
    """学習と同じ入力を**さらに短く**した評価仕様。ここで見たいのは保存/読込の
    往復であって波形の質ではないので、シミュ長は最小で足りる。"""
    ds = bundle.meta.dataset
    return {
        "hh_dc": {
            "spec": SimSpec(
                target=ds.model_name,
                current_type=ds.current_type,
                dt=ds.dt,
                current_params={**ds.current_params, "duration": 170.0},
            )
        }
    }


def test_eval_runs_round_trip_without_resimulating(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """1 run = 1 SimSpec。保存 → 読込で再シミュ無しに同じ波形が戻り、
    `(系列名, 点 index, run_id)` ごとに分かれた run が読込で束ね直る。同じ入力の再実行は
    スキップされ、原系 (親 run) は学習 run を跨いで共有される。"""
    evals = _evals(sindy)
    logged = mlflow_io.run_and_log({RUN_ID: sindy}, evals, RUN_ID)
    assert len(logged) == 1  # 返すのは子 (置換系) だけ = 原系は親として 1 本

    loaded = mlflow_io.load_eval_results([RUN_ID])
    assert set(loaded) == {("hh_dc", 0, None), ("hh_dc", 0, RUN_ID)}
    # 入力仕様が往復し、そこから dataset を復元できる
    assert (
        loaded[("hh_dc", 0, RUN_ID)].spec.dataset().model_name
        == evals["hh_dc"]["spec"].target
    )
    # 単発 (掃引軸なし) は軸が無いまま戻る (param は文字列なので "None" にならない)
    assert loaded[("hh_dc", 0, None)].axis is None
    assert loaded[("hh_dc", 0, None)].point is None
    # 波形は float32 で往復し、原系/置換系はそれぞれの run に入っている
    # (原系は surrogate 非依存なので回し直しても一致する)
    np.testing.assert_allclose(
        access.potential(loaded[("hh_dc", 0, None)].dataset, 0),
        access.potential(simulate(evals["hh_dc"]["spec"], None).dataset, 0),
        rtol=1e-5,
    )
    assert not np.allclose(
        access.potential(loaded[("hh_dc", 0, RUN_ID)].dataset, 0),
        access.potential(loaded[("hh_dc", 0, None)].dataset, 0),
    )

    # シミュは決定的 → 同じ入力の 2 度目はスキップ (spec_hash 一致)
    assert mlflow_io.run_and_log({RUN_ID: sindy}, evals, RUN_ID) == []

    # 別の学習 run から同じ条件 → 子は増えるが原系 (親) は共有される
    # (学習 run を増やしても原系の波形が複製されない)
    mlflow_io.run_and_log({"OTHER": sindy}, evals, "OTHER")
    originals = mlflow.search_runs(
        experiment_ids=[mlflow_io._eval_exp_id()],
        filter_string="tags.kind = 'original'",
        output_format="list",
    )
    assert len(originals) == 1
    both = mlflow_io.load_eval_results([RUN_ID, "OTHER"])
    assert set(both) == {
        ("hh_dc", 0, None),
        ("hh_dc", 0, RUN_ID),
        ("hh_dc", 0, "OTHER"),
    }
