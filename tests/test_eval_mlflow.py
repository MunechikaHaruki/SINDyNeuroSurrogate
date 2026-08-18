"""評価結果の永続化 (MLflow の評価 experiment) の smoke。

`tests/test_surrogate.py` がドメイン層だけを通すのに対し、ここは
`scripts/mlflow_io/` (experiment と run、成果物の段名と書き出し) を
通す。tracking 先は tmp の sqlite へ丸ごと差し替えるので、手元の `mlflow.db` /
`mlruns/` は汚れない。
"""

import json
from pathlib import Path
from typing import cast

import mlflow

# `scripts/` は conftest が import path へ入れている。
import mlflow_io.artifacts as artifacts_io  # noqa: E402
import mlflow_io.report as report_io  # noqa: E402
import mlflow_io.series as series_io  # noqa: E402
import numpy as np
import pytest
from mlflow.entities import Run
from mlflow_io.surrogate import log_surrogate_model
from test_surrogate import fit_surrogate

from neurosurrogate.artifact.model import Tuning
from neurosurrogate.core import access
from neurosurrogate.sim.run import simulate
from neurosurrogate.sim.spec import EvalSelection, EvalSeries
from neurosurrogate.surrogate.bundle import SurrogateBundle

RUN_ID = "RID"  # 学習 run の代役 (評価 run が指す先)


def _exp_id(name: str) -> str:
    """experiment 名 → id。本番の解決子 (`mlflow_io._query`) はパッケージ内専用なので、
    テストは MLflow を直に引く。"""
    exp = mlflow.get_experiment_by_name(name)
    assert exp is not None
    return str(exp.experiment_id)


def _by_hash(exp_name: str, tag: str, value: str) -> Run | None:
    """同一性の鍵で run を引く。**tag の綴りはテストが直書きする** — 本番の定数を
    借りると綴りを変えても落ちず、保存の契約を検査したことにならない (鍵の組み方の
    ようなロジックは逆に本番のものを呼ぶ)。"""
    return next(
        iter(
            cast(
                list[Run],
                mlflow.search_runs(
                    experiment_ids=[_exp_id(exp_name)],
                    filter_string=f"tags.{tag} = '{value}'",
                    output_format="list",
                ),
            )
        ),
        None,
    )


def _source_runs(report_run_id: str) -> tuple[str, ...]:
    """レポート run が指す波形 run (原系が先)。**本番は使わない** — `load_report` は
    描く中身だけを返し、由来の id は描画にも保存段にも要らない。"""
    tags = mlflow.get_run(report_run_id).data.tags
    return (
        tags["original_series_id"],
        *json.loads(tags["surrogate_series_ids"]),
    )


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
    カタログも差し替える — `run_report`/`find_report_run` は系列を**名前から引く**の
    で、テスト用の短い掃引 (`_evals`) をその名前に載せて渡す。"""
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
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """**1 波形 run = 1 EvalSeries、1 レポート run = 1 回の評価**。掃引点の波形を
    まとめた 1 artifact が往復し、再シミュ無しに点の並びごと戻る。同じ掃引の再実行は
    スキップされ、原系の run は学習 run を跨いで共有される。レポートは
    **1 系列 × N モデル**が単位で、選択 (学習 run 群 × 系列 1 つ) で引き、波形 run
    への参照だけを持つ。"""
    series = _evals(sindy)
    report_id = report_io.run_report({RUN_ID: sindy}, "hh_dc")
    # 波形は 2 本 (原系 1 + 置換系 1)、レポートは 1 本 (点では分かれない)
    assert len(_of_kind("original")) == 1 and len(_of_kind("surrogate")) == 1
    # 選択 → レポート run_id の橋は 1 本で、描画はこの id だけを見る
    assert report_io.find_report_run("hh_dc", (RUN_ID,)) == report_id

    view = report_io.load_report(report_id)
    assert view.run_ids == [RUN_ID]
    # 結果から作り直した記述で同じレポートに当たる = 記述 (`EvalSelection`) と
    # 結果 (`SeriesResults`) が同じ対で往復する。**往復するのは選択した run が全部
    # 置換できたときだけ** — 鍵は選択そのもの (`EvalSelection` の docstring) で、
    # 結果に並ぶのは置換できた列だけだから。
    restored = _by_hash(
        report_io.REPORT_EXP,
        "report_hash",
        EvalSelection(view.series, tuple(view.run_ids)).hash(),
    )
    assert restored is not None and restored.info.run_id == report_id
    assert (
        EvalSelection(series, (RUN_ID,)).hash()
        == EvalSelection(view.series, tuple(view.run_ids)).hash()
    )
    # 系列名は結果 (`SeriesResults`) でなく波形 run 側にある = レポートはカタログを
    # 参照しない (名前を付け替えても過去のレポートは読める)
    original_eval, surr_eval = _source_runs(report_id)
    assert mlflow.get_run(original_eval).data.params["name"] == "hh_dc"
    assert series_io._series_hash(series, "abcdefgh-1") != series_io._series_hash(
        series, "abcdefgh-2"
    )
    # 点は宣言した掃引値の順で戻る (点ごとの識別子を保存していない)
    assert (view.axis, view.values) == ("duration", [170.0, 190.0])
    orig, surr = view.pair(1, view.column(RUN_ID))
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

    # シミュは決定的 → 同じ掃引の 2 度目はスキップ (series_hash 一致)。同じ選択の
    # レポートも量産されず、同じ run が更新される。
    assert report_io.run_report({RUN_ID: sindy}, "hh_dc") == report_id
    assert len(_of_kind("surrogate")) == 1

    # 別の学習 run から同じ条件 → 置換系は増えるが原系は共有される
    # (学習 run を増やしても原系の波形が複製されない)。選択が違えば別のレポート。
    assert report_io.run_report({"OTHER": sindy}, "hh_dc") != report_id
    assert (len(_of_kind("original")), len(_of_kind("surrogate"))) == (1, 2)

    # 学習 run 2 件の選択はさらに別のレポート = 別の単位 (run 軸 2 本が 1 枚に並ぶ)
    both = report_io.load_report(
        report_io.run_report({RUN_ID: sindy, "OTHER": sindy}, "hh_dc")
    )
    assert (len(both.original_waves), both.run_ids) == (2, [RUN_ID, "OTHER"])


def test_unevaluated_selection_finds_nothing_without_creating_experiment(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """まだ評価していない選択は `None` が返るだけ (描画側は「先に評価」を出す)。
    探すだけの経路は experiment を作らない = 選択を変えるたびに空の experiment が
    生えない。"""
    assert report_io.find_report_run("hh_dc", (RUN_ID,)) is None
    assert mlflow.get_experiment_by_name(report_io.REPORT_EXP) is None


def test_run_report_rejects_series_no_run_can_replace(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """1 本も置換できない選択は回す意味が無い → 空のレポートを作らず落ちる。"""
    with pytest.raises(ValueError, match="置換できない"):
        report_io.run_report({}, "hh_dc")


def test_run_directories_disambiguate_after_slugging(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """異なる run 名が path 安全化後に同じ綴りでも、保存段は衝突しない。"""
    first = _train_run("a/b", sindy)
    second = _train_run("a:b", sindy)
    directories = artifacts_io._run_dirs([first, second])
    assert len(set(directories.values())) == 2
    assert all(name.startswith("a-b-") for name in directories.values())


def test_everything_drawn_lands_in_the_one_report_run(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """描画の入力は**レポート run_id 1 つ + `Tuning`** だけで、描いたものは全部その
    レポート run の artifact になる (比べたいのが 1 系列 × N モデルの束そのものなので、
    束ねる単位はレポート以外に無い)。**学習 run にも波形 run にも書かない** = 記録した
    run を描画が書き換えない。

    `eval_comp` が適用先に無ければ設定誤りとして描画を失敗させる。
    """
    # 段名は学習 run の MLflow run 名なので、代役 id でなく実在の学習 run を立てる。
    train_id = _train_run("surr-A", sindy)
    report_id = report_io.run_report({train_id: sindy}, "hh_dc")
    view = report_io.load_report(report_id)
    written = artifacts_io.log_report_artifacts(
        report_id, Tuning(eval_comp=view.net.names[0])
    )
    # run 内の名前は元の 3 段のまま: models/<run 名>/ は比べた 1 本ずつの自己記述図、
    # series/<run 名>/ は波形 1 本で決まるもの、直下が run 横断の産物。
    assert {w.split("/")[0] for w in written} == {
        "models",
        "series",
        "summary.csv",
        "traces.png",
        "metric.png",  # 掃引 (2 点) なので点軸の折れ線も出る
        "draw.json",  # 描いたときの `Tuning` も返りに載る (返り = その run の artifact)
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
    assert "draw.json" in {a.path for a in client.list_artifacts(report_id)}

    with pytest.raises(ValueError, match="eval_comp"):
        artifacts_io.log_report_artifacts(report_id, Tuning(eval_comp="nope"))


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
