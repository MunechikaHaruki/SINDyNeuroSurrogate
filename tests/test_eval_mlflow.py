"""評価結果の永続化 (MLflow の評価 experiment) の smoke。

`tests/test_surrogate.py` がドメイン層だけを通すのに対し、ここは
`scripts/mlflow_io/` (experiment と run、成果物の段名と書き出し) を
通す。tracking 先は tmp の sqlite へ丸ごと差し替えるので、手元の `mlflow.db` /
`mlruns/` は汚れない。
"""

import json
from pathlib import Path
from typing import Any, cast

import mlflow

# `scripts/` は conftest が import path へ入れている。
import mlflow_io.report as report_io  # noqa: E402
import mlflow_io.series as series_io  # noqa: E402
import numpy as np
import pytest
from mlflow.entities import Run
from mlflow_io.surrogate import log_surrogate_model
from test_surrogate import fit_surrogate

from neurosurrogate.core import access
from neurosurrogate.sim.result import SeriesResults
from neurosurrogate.sim.run import simulate
from neurosurrogate.sim.spec import EvalSeries
from neurosurrogate.surrogate.bundle import SurrogateBundle

RUN_NAME = "RID"


def _exp_id(name: str) -> str:
    """experiment 名 → id。本番の解決子 (`mlflow_io._query`) はパッケージ内専用なので、
    テストは MLflow を直に引く。"""
    exp = mlflow.get_experiment_by_name(name)
    assert exp is not None
    return str(exp.experiment_id)


def _source_runs(report_run_id: str) -> tuple[str, ...]:
    """レポート run が指す波形 run (原系が先)。**本番は使わない**。"""
    tags = mlflow.get_run(report_run_id).data.tags
    return (
        tags["original_series_id"],
        *json.loads(tags["surrogate_series_ids"]),
    )


def _view(report_run_id: str) -> SeriesResults:
    """保存内容の検査用に、レポートが指す波形列を組み立てる。"""
    source_runs = _source_runs(report_run_id)
    return SeriesResults(
        series_io.load_column(source_runs[0]),
        tuple(series_io.load_column(run_id) for run_id in source_runs[1:]),
    )


def _report_ids() -> set[str]:
    """副作用だけの `write_report` が保存した report run の集合。"""
    exp = mlflow.get_experiment_by_name(report_io.REPORT_EXP)
    if exp is None:
        return set()
    return _ids(
        cast(
            list[Run],
            mlflow.search_runs(
                experiment_ids=[exp.experiment_id], output_format="list"
            ),
        )
    )


def _write_report(
    run_ids: tuple[str, ...], tuning: dict[str, Any] | None = None
) -> str:
    """戻り値に頼らず、呼出前後の差分から新しい report run を一意に得る。"""
    before = _report_ids()
    assert report_io.write_report(run_ids, "hh_dc", tuning or _tuning()) is None
    after = _report_ids()
    assert before < after
    created = after - before
    assert len(created) == 1
    return created.pop()


def _artifact_paths(run_id: str) -> set[str]:
    """MLflow run 以下のファイルを再帰的に列挙する。"""
    client = mlflow.MlflowClient()
    pending = [""]
    paths: set[str] = set()
    while pending:
        for artifact in client.list_artifacts(run_id, pending.pop()):
            if artifact.is_dir:
                pending.append(artifact.path)
            else:
                paths.add(artifact.path)
    return paths


def _tuning(
    eval_comp: str = "soma", detail: dict[str, int] | None = None
) -> dict[str, Any]:
    return {
        "common": {"eval_comp": eval_comp, "view_comps": []},
        "report": {},
        "detail": detail or {},
    }


def _train_run(name: str, bundle: SurrogateBundle) -> str:
    """名前と surrogate artifact を持つ学習 run を 1 本立てる。"""
    exp = mlflow.get_experiment_by_name("test_train")
    experiment_id = exp.experiment_id if exp else mlflow.create_experiment("test_train")
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=name,
    ) as run:
        log_surrogate_model(bundle)
        return str(run.info.run_id)


@pytest.fixture(scope="module")
def sindy() -> SurrogateBundle:
    return fit_surrogate("_test_hh_sindy")


@pytest.fixture
def eval_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sindy: SurrogateBundle
) -> str:
    """tracking 先を tmp へ移し、評価/レポート experiment 名もテスト専用にする。
    カタログも差し替える — `write_report` は系列を**名前から引く**ので、テスト用の
    短い掃引 (`_evals`) をその名前に載せて渡す。"""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(series_io, "EVAL_EXP", "test_eval")
    monkeypatch.setattr(report_io, "REPORT_EXP", "test_report")
    monkeypatch.setattr(report_io, "SERIES", {"hh_dc": _evals(sindy)})
    return "test_eval"


def _evals(bundle: SurrogateBundle) -> EvalSeries:
    """学習と同じ入力を**さらに短く**した評価仕様 (2 点の掃引)。ここで見たいのは
    保存/読込の往復であって波形の質ではないので、シミュ長は最小で足りる。掃引に
    してあるのは、点の並びが `EvalSeries.points` から復元されることを見るため。"""
    return EvalSeries(spec=bundle.meta.dataset, param="duration", values=[170.0, 190.0])


def test_eval_runs_round_trip_without_resimulating(
    eval_store: str, sindy: SurrogateBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**1 波形 run = 1 EvalSeries、1 レポート run = 1 回の評価**。掃引点の波形を
    まとめた 1 artifact が往復し、再シミュ無しに点の並びごと戻る。同じ掃引の再実行は
    スキップされ、原系の run は学習 run を跨いで共有される。レポートは
    **1 系列 × N モデル**が単位で、波形 run への参照だけを持つ。"""
    series = _evals(sindy)
    train_id = _train_run(RUN_NAME, sindy)
    # このテストは波形/run の往復を見る。重い描画は次の統合テスト 1 件に任せる。
    monkeypatch.setattr(report_io, "save_report", lambda *_: None)
    report_id = _write_report((train_id,))
    # 波形は 2 本 (原系 1 + 置換系 1)、レポートは 1 本 (点では分かれない)
    assert len(_of_kind("original")) == 1 and len(_of_kind("surrogate")) == 1

    view = _view(report_id)
    assert view.run_ids == [train_id]
    # 掃引の記述は結果から作り直せる (記述と結果が同じ対で往復する)
    assert view.series.hash() == series.hash()
    original_eval, surr_eval = _source_runs(report_id)
    # 点は宣言した掃引値の順で戻る (点ごとの識別子を保存していない)
    assert (view.series.param, view.series.axis_values) == ("duration", [170.0, 190.0])
    orig, surr = view.pair(1, view.column(train_id))
    # 記述が往復し、点の計算入力からそのまま実行入力を復元できる
    # (波形は並びだけで点に対応する = 点ごとの仕様は保存しない)
    assert view.series.points[1].materialize().net is series.spec.net
    # 由来 (どの評価 run から読んだか) はレポート run の tag だけが持つ (原系 1 本 +
    # 置換系 1 本)。描く中身にも保存段にも出てこない。
    assert original_eval in _ids(_of_kind("original"))
    assert surr_eval in _ids(_of_kind("surrogate"))
    # 波形は float32 で往復し、原系/置換系はそれぞれの run に入っている
    # (原系は surrogate 非依存なので回し直しても一致する)
    np.testing.assert_allclose(
        access.potential(orig, 0),
        access.potential(simulate(series.points[1], None), 0),
        rtol=1e-5,
    )
    assert not np.allclose(access.potential(surr, 0), access.potential(orig, 0))

    # シミュは決定的 → 同じ掃引の 2 度目はスキップ (series_hash 一致)。**レポートは
    # 押すたびに新しい run** で、既に書いた run は書き換わらない (重いのは波形だけ)。
    again = _write_report((train_id,))
    assert again != report_id
    assert _source_runs(again) == _source_runs(report_id)
    assert len(_of_kind("surrogate")) == 1

    # 別の学習 run から同じ条件 → 置換系は増えるが原系は共有される
    # (学習 run を増やしても原系の波形が複製されない)。
    other_id = _train_run("OTHER", sindy)
    _write_report((other_id,))
    assert (len(_of_kind("original")), len(_of_kind("surrogate"))) == (1, 2)

    # 学習 run 2 件の選択は run 軸 2 本が 1 枚に並ぶ
    both_id = _write_report((train_id, other_id))
    both = _view(both_id)
    assert (len(both.original_waves), both.run_ids) == (2, [train_id, other_id])


def test_write_report_rejects_series_no_run_can_replace(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """1 本も置換できない選択は回す意味が無い → 空のレポートを作らず落ちる。"""
    with pytest.raises(ValueError, match="置換できない"):
        report_io.write_report((), "hh_dc", _tuning())


def test_everything_drawn_lands_in_the_one_report_run(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """描画の入力は**レポート run_id 1 つ + つまみ**だけで、描いたものは全部その
    レポート run の artifact になる (比べたいのが 1 系列 × N モデルの束そのものなので、
    束ねる単位はレポート以外に無い)。**学習 run にも波形 run にも書かない** = 記録した
    run を描画が書き換えない。

    `eval_comp` が適用先に無ければ設定誤りとして描画を失敗させる。
    """
    # 段名は学習 run の MLflow run 名なので、代役 id でなく実在の学習 run を立てる。
    train_id = _train_run("surr-A", sindy)
    report_id = _write_report((train_id,))
    written = _artifact_paths(report_id)
    # run 内の名前は元の 3 段のまま: models/<run 名>/ は比べた 1 本ずつの自己記述図、
    # series/<run 名>/ は波形 1 本で決まるもの、直下が run 横断の産物。
    assert {w.split("/")[0] for w in written} == {
        "models",
        "series",
        "summary.csv",
        "traces.png",
        "metric.png",  # 掃引 (2 点) なので点軸の折れ線も出る
        # 描いたときのつまみも同じ run の artifact に載る
        "tuning.json",
    }
    # 段名は学習 run の MLflow run 名 (models/ と series/ で同じ綴り)
    name = "surr-A"
    assert f"models/{name}/model.png" in written
    assert any(w.startswith(f"series/{name}/") for w in written)
    assert "series/original/current.png" in written

    # 書けたのはレポート run だけ (学習 run / 波形 run の artifact は増えない)
    client = mlflow.MlflowClient()
    assert {a.path for a in client.list_artifacts(report_id)} >= {"models", "series"}
    for rid in _source_runs(report_id):
        assert [a.path for a in client.list_artifacts(rid)] == [series_io.WAVES_FILE]
    # 描いたときの表示設定は 1 枚だけ添える (成果物ごとの由来は run が既に指している)
    assert "tuning.json" in {a.path for a in client.list_artifacts(report_id)}

    report_ids = _report_ids()
    with pytest.raises(ValueError, match="eval_comp"):
        report_io.write_report((train_id,), "hh_dc", _tuning("nope"))
    assert _report_ids() == report_ids

    # 点軸の外を指した詳細図も設定誤り (端へ丸めて別の点の図を黙って出さない)
    with pytest.raises(ValueError, match="点 index"):
        report_io.write_report(
            (train_id,), "hh_dc", _tuning(detail={"detail_point": 9})
        )
    assert _report_ids() == report_ids


def _ids(runs: list[Run]) -> set[str]:
    return {r.info.run_id for r in runs}


def _of_kind(kind: str) -> list[Run]:
    """波形 experiment の run を kind で数える (原系が複製されないことの確認用)。"""
    return cast(
        list[Run],
        mlflow.search_runs(
            experiment_ids=[_exp_id(series_io.EVAL_EXP)],
            filter_string=f"tags.kind = '{kind}'",
            output_format="list",
        ),
    )
