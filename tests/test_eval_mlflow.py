"""評価結果の永続化 (MLflow の評価 experiment) の smoke。

`tests/test_surrogate.py` がドメイン層だけを通すのに対し、ここは
`scripts/mlflow_io/` = **MLflow を知る唯一の場所**を通す。tracking 先は tmp の
sqlite へ丸ごと差し替えるので、手元の `mlflow.db` / `mlruns/` は汚れない。
"""

from pathlib import Path
from typing import cast

import mlflow

# `scripts/` は conftest が import path へ入れている。
import mlflow_io.report as report_io  # noqa: E402
import mlflow_io.save as save  # noqa: E402
import mlflow_io.series as series_io  # noqa: E402
import numpy as np
import pytest
from mlflow.entities import Run
from mlflow_io.report import report_artifacts, series_artifacts
from mlflow_io.save import save_artifacts, slug
from mlflow_io.surrogate import model_artifacts
from test_surrogate import fit_surrogate
from tuning import Tuning

from neurosurrogate.core import access
from neurosurrogate.sim.figures import run_names
from neurosurrogate.sim.run import simulate
from neurosurrogate.sim.spec import EvalSeries
from neurosurrogate.surrogate.bundle import SurrogateBundle

RUN_ID = "RID"  # 学習 run の代役 (評価 run が指す先)


@pytest.fixture
def eval_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """tracking 先を tmp へ移し、評価/レポート experiment 名もテスト専用にする。"""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(series_io, "EVAL_EXP", "test_eval")
    monkeypatch.setattr(report_io, "REPORT_EXP", "test_report")
    return "test_eval"


@pytest.fixture(scope="module")
def sindy() -> SurrogateBundle:
    return fit_surrogate("_test_hh_sindy")


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
    report_id = report_io.run_and_log({RUN_ID: sindy}, "hh_dc", series)
    # 波形は 2 本 (原系 1 + 置換系 1)、レポートは 1 本 (点では分かれない)
    assert len(_of_kind("original")) == 1 and len(_of_kind("surrogate")) == 1
    # 選択 → レポート run_id の橋は 1 本で、描画はこの id だけを見る
    assert report_io.find_report_run([RUN_ID], series) == report_id

    report = report_io.load_report(report_id)
    view = report.view
    assert view.run_ids == [RUN_ID]
    # 系列名は結果 (`SeriesResults`) でなく波形 run 側にある = レポートはカタログを
    # 参照しない (名前を付け替えても過去のレポートは読める)
    assert series_io.name_of(report.original_id) == "hh_dc"
    # 点は宣言した掃引値の順で戻る (点ごとの識別子を保存していない)
    assert (view.axis, view.values) == ("duration", [170.0, 190.0])
    orig, surr = view.pair(1, RUN_ID)
    # 入力仕様が往復し、そこから実行入力を復元できる
    assert surr.spec.materialize().net is series.spec.net
    # 由来 (どの評価 run から読んだか) は描く中身 (`SeriesResults`) でなく `Report` 側が
    # 持つ (原系 1 本 + 置換系 1 本)。置換系は**学習 run → 評価 run** の対応で持つので、
    # 保存段も凡例も id 1 つから解ける。
    assert report.original_id in _ids(_of_kind("original"))
    assert report.surr_ids[RUN_ID] in _ids(_of_kind("surrogate"))
    assert report.sources == (report.original_id, report.surr_ids[RUN_ID])
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
    assert report_io.run_and_log({RUN_ID: sindy}, "hh_dc", series) == report_id
    assert len(_of_kind("surrogate")) == 1

    # 別の学習 run から同じ条件 → 置換系は増えるが原系は共有される
    # (学習 run を増やしても原系の波形が複製されない)。選択が違えば別のレポート。
    assert report_io.run_and_log({"OTHER": sindy}, "hh_dc", series) != report_id
    assert (len(_of_kind("original")), len(_of_kind("surrogate"))) == (1, 2)

    # 学習 run 2 件の選択はさらに別のレポート = 別の単位 (run 軸 2 本が 1 枚に並ぶ)
    both = report_io.load_report(
        report_io.run_and_log({RUN_ID: sindy, "OTHER": sindy}, "hh_dc", series)
    )
    assert (len(both.view.points), both.view.run_ids) == (2, [RUN_ID, "OTHER"])


def test_report_rejects_ids_that_do_not_line_up_with_the_run_axis(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """`Report` は id が **run 軸と揃う**ことを構築時に保証する。半端に欠けた id は
    保存段が別の run の名前へ落ちる (由来の欠けた成果物) ので、描く前に弾く。"""
    view = report_io.load_report(
        report_io.run_and_log({RUN_ID: sindy}, "hh_dc", _evals(sindy))
    ).view
    for original_id, surr_ids in (
        ("e0", {"other": "e1"}),  # run 軸とキーがずれる
        ("e0", {}),  # 置換系の id が丸ごと無い
        ("", {RUN_ID: "e1"}),  # 原系の id が無い
        ("e0", {RUN_ID: ""}),  # 空文字の id も欠けと同じ
        ("", {RUN_ID: ""}),
    ):
        with pytest.raises(ValueError, match="評価 run の id"):
            report_io.Report("r", view, original_id, surr_ids)


def test_unevaluated_selection_finds_nothing_without_creating_experiment(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """まだ評価していない選択は `None` が返るだけ (描画側は「先に評価」を出す)。
    探すだけの経路は experiment を作らない = 選択を変えるたびに空の experiment が
    生えない。"""
    assert report_io.find_report_run([RUN_ID], _evals(sindy)) is None
    assert mlflow.get_experiment_by_name(report_io.REPORT_EXP) is None


def test_run_and_log_rejects_series_no_run_can_replace(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """1 本も置換できない選択は回す意味が無い → 空のレポートを作らず落ちる。"""
    with pytest.raises(ValueError, match="置換できない"):
        report_io.run_and_log({}, "hh_dc", _evals(sindy))


def test_everything_drawn_lands_in_the_one_report_run(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """描画の入力は**レポート run_id 1 つ + `Tuning`** だけで、描いたものは全部その
    レポート run の artifact になる (比べたいのが 1 系列 × N モデルの束そのものなので、
    束ねる単位はレポート以外に無い)。**学習 run にも波形 run にも書かない** = 記録した
    run を描画が書き換えない。

    `eval_comp` が適用先に無ければ黙って描かずエラー図 1 枚 (誤りも同じレポートの中)。
    """
    report_id = report_io.run_and_log({RUN_ID: sindy}, "hh_dc", _evals(sindy))
    report = report_io.load_report(report_id)
    bundles = {RUN_ID: sindy}
    tuning = Tuning(eval_comp=report.view.net.names[0])
    written = save_artifacts(
        [
            *model_artifacts(bundles, tuning),
            *series_artifacts(report, bundles, tuning),
            *report_artifacts(report, bundles, tuning),
        ],
        report_id,
        tuning,
    )
    # run 内の名前は元の 3 段のまま: models/<表示名>/ は比べた 1 本ずつの自己記述図、
    # series/<表示名>/ は波形 1 本で決まるもの、直下が run 横断の産物。
    assert {w.split("/")[0] for w in written} == {
        "models",
        "series",
        "summary.csv",
        "traces.png",
        "metric.png",  # 掃引 (2 点) なので点軸の折れ線も出る
    }
    label = slug(run_names(bundles)[RUN_ID])
    assert f"models/{label}/model.png" in written
    assert "series/original/current.png" in written

    # 書けたのはレポート run だけ (学習 run / 波形 run の artifact は増えない)
    client = mlflow.MlflowClient()
    assert {a.path for a in client.list_artifacts(report_id)} >= {"models", "series"}
    for rid in report.sources:
        assert [a.path for a in client.list_artifacts(rid)] == [series_io.WAVES_FILE]
    # 描いたときの表示設定は 1 枚だけ添える (成果物ごとの由来は run が既に指している)
    assert save.DRAW_FILE in {a.path for a in client.list_artifacts(report_id)}

    # 適用先に無い comp: 波形を見る図はエラー図 1 枚に畳む (波形を見ないサマリ表は
    # そのまま出る)。
    err = report_artifacts(report, bundles, Tuning(eval_comp="nope"))
    assert "error" in {a.name for a in err}
    assert not any(a.name in ("traces", "metric") for a in err)


def _ids(runs: list[Run]) -> set[str]:
    return {r.info.run_id for r in runs}


def _of_kind(kind: str) -> list[Run]:
    """波形 experiment の run を kind で数える (原系が複製されないことの確認用)。"""
    return cast(
        list[Run],
        mlflow.search_runs(
            experiment_ids=[series_io._eval_exp_id()],
            filter_string=f"tags.kind = '{kind}'",
            output_format="list",
        ),
    )
