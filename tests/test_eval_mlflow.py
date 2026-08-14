"""評価結果の永続化 (MLflow の評価 experiment) の smoke。

`tests/test_surrogate.py` がドメイン層だけを通すのに対し、ここは
`scripts/mlflow_io/` = **MLflow を知る唯一の場所**を通す。tracking 先は tmp の
sqlite へ丸ごと差し替えるので、手元の `mlflow.db` / `mlruns/` は汚れない。
"""

import json
from pathlib import Path
from typing import cast

import mlflow

# `scripts/` は conftest が import path へ入れている。
import mlflow_io.report as report_io  # noqa: E402
import mlflow_io.series as series_io  # noqa: E402
import numpy as np
import pandas as pd
import pytest
from mlflow.entities import Run
from mlflow_io.report import report_entries
from mlflow_io.save import SaveEntry, save_entries, slug
from mlflow_io.series import series_entries
from mlflow_io.surrogate import model_entries
from test_surrogate import fit_surrogate
from tuning import Tuning

from neurosurrogate.core import access
from neurosurrogate.plotting import Artifact, new_figure
from neurosurrogate.sim.eval import EvalSeries, simulate
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

    view = report_io.load_report(report_id)
    assert view.name == "hh_dc"
    assert view.run_ids == [RUN_ID]
    # 点は宣言した掃引値の順で戻る (点ごとの識別子を保存していない)
    assert (view.axis, view.values) == ("duration", [170.0, 190.0])
    orig, surr = view.pair(1, RUN_ID)
    # 入力仕様が往復し、そこから実行入力を復元できる
    assert surr.spec.materialize().net is series.spec.net
    # 由来 (どの評価 run から読んだか) は結果でなく `SeriesView` 側が持つ
    # (原系 1 本 + 置換系 1 本)。置換系は**学習 run → 評価 run** の対応で持つので、
    # 保存段も凡例も id 1 つから解ける。
    assert view.original_id in _ids(_of_kind("original"))
    assert view.surr_ids[RUN_ID] in _ids(_of_kind("surrogate"))
    assert view.sources == (view.original_id, view.surr_ids[RUN_ID])
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
    assert (len(both.points), both.run_ids) == (2, [RUN_ID, "OTHER"])


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


def test_resolved_run_ids_draw_independent_entry_groups(
    eval_store: str, sindy: SurrogateBundle
) -> None:
    """描画の入力は**レポート run_id 1 つ + `Tuning`** だけ。surrogate は波形に
    焼き込まれていないので run_id から引き直す (ここでは学習 run が代役なので差し替え)。
    `eval_comp` が適用先に無ければ黙って描かずエラー図 1 枚になる (誤りは選んだ
    レポートに紐づくのでレポート配下 = 別レポートのエラー図と潰し合わない)。"""
    report_id = report_io.run_and_log({RUN_ID: sindy}, "hh_dc", _evals(sindy))
    comp = report_io.load_report(report_id).net.names[0]

    view = report_io.load_report(report_id)
    bundles = {RUN_ID: sindy}
    tuning = Tuning(eval_comp=comp)
    entries = [
        *model_entries(bundles, tuning),
        *series_entries(view, bundles, tuning),
        *report_entries(view, bundles, tuning, report_id),
    ]
    assert len(entries) > 1 and all(e.artifact.obj is not None for e in entries)
    # 保存名は MLflow の 3 experiment がそのまま 3 段: models/<学習 run>/ は描く対象が
    # 学習 run そのもの (レポートを増やしても複製されない)、series/<評価 run>/ は
    # 波形 1 本で決まるもの (入力電流と詳細図)、report/<レポート run>/ は run 横断の
    # 産物。段の名前は MLflow の run 名。
    assert {e.stage.split("/")[0] for e in entries} == {"models", "series", "report"}
    # 段の名前 = run 名 + run id 先頭。**名前だけでは一意でない** (人が付け替えられ、
    # 掃引違いの評価 run は同名になる) ので、id を混ぜて別 run が潰し合わないように
    # する。学習 run は代役 (実在しない id) なので名前が引けず id 自身が名前に落ちる。
    assert {e.stage.split("/")[1] for e in entries} == {
        f"{RUN_ID}-{RUN_ID[:8]}",
        *(
            f"{slug(mlflow.get_run(rid).info.run_name)}-{rid[:8]}"
            for rid in (report_id, *view.sources)
        ),
    }

    # 適用先に無い comp: 波形を見る図はエラー図 1 枚に畳む (波形を見ないサマリ表は
    # そのまま出る)。誤りは選んだレポートに紐づくのでレポート配下 = 別レポートの
    # エラー図と潰し合わない。
    err = report_entries(view, bundles, Tuning(eval_comp="nope"), report_id)
    report_dir = f"{slug(mlflow.get_run(report_id).info.run_name)}-{report_id[:8]}"
    assert f"report/{report_dir}/error.png" in {e.path for e in err}
    assert not any(e.artifact.name in ("traces", "metric") for e in err)


def test_second_report_saved_into_the_same_dest_keeps_the_first(
    tmp_path: Path,
) -> None:
    """`results/` 直下が全レポート共通の dest (レポートごとに dest を割らない):
    名前が `report/<レポート run>/...` で割れているのでファイルは潰し合わず、
    `meta.json` も合流して前のレポートの由来が残る。`models/` は同じパスへ上書き。"""
    save_entries([SaveEntry("report/r1", Artifact("traces", new_figure()))], tmp_path)
    save_entries(
        [
            SaveEntry("report/r2", Artifact("traces", new_figure())),
            SaveEntry("models", Artifact("m", pd.DataFrame({"a": [1]}))),
        ],
        tmp_path,
    )

    assert (tmp_path / "report/r1/traces.png").exists()
    assert (tmp_path / "report/r2/traces.png").exists()
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert set(meta) == {"report/r1/traces.png", "report/r2/traces.png", "models/m.csv"}


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
