"""サロゲート fit → 置換シミュ → 指標/描画の smoke (marimo/MLflow 非依存)。

Hydra プリセットを実設定源として読み、UI/実験ログを介さずドメイン層だけを通す。
設定は `conf/surrogate/_test_*.yaml` (素体から library_specs を継承し、学習構造と
短縮電流だけ固定したテスト専用プリセット) に置き、テスト側は override しない。
"""

from dataclasses import replace as dc_replace
from functools import cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from catalog import SERIES
from hydra import compose, initialize_config_dir
from matplotlib.figure import Figure
from omegaconf import OmegaConf

import neurosurrogate.artifact.bundle as artifact_bundle
from neurosurrogate.artifact.bundle import (
    detail_artifacts,
    original_artifacts,
    report_artifacts,
    surrogate_artifacts,
)
from neurosurrogate.core import access
from neurosurrogate.core.coords import transform_gate
from neurosurrogate.core.opcost import OpCost
from neurosurrogate.core.simulator import unified_simulator
from neurosurrogate.neurons.compartments.hh import HHParams, dhdt, dmdt, dndt, hh_inits
from neurosurrogate.neurons.compartments.traub import (
    TRAUB_EXTRA_GATE_NAMES,
    TRAUB_SR_EXTRA_GATE_NAMES,
)
from neurosurrogate.sim.artifacts import summary_artifact, traces_artifact
from neurosurrogate.sim.result import SeriesResults, SeriesRun
from neurosurrogate.sim.run import run_column
from neurosurrogate.sim.spec import EvalSeries, SimSpec
from neurosurrogate.surrogate.artifacts.model import (
    feature_tex,
    preprocessor_artifact,
    tex,
)
from neurosurrogate.surrogate.artifacts.train import (
    train_manifold_artifact,
    train_preprocessed_artifact,
    train_raw_artifact,
    train_recon_artifact,
    train_v_coverage_artifact,
)
from neurosurrogate.surrogate.model import Surrogate
from neurosurrogate.surrogate.parts import Closure, Preprocessor
from neurosurrogate.surrogate.parts.ansatz.hybrid import (
    HYBRID_PHYSICS,
    HybridAnsatz,
)
from neurosurrogate.surrogate.parts.ansatz.ude import UDEAnsatz
from neurosurrogate.surrogate.parts.closure.sindy import SINDyBundle
from neurosurrogate.surrogate.parts.closure.sindy.entry import FeatureLibrary
from neurosurrogate.surrogate.parts.closure.ude import UDEClosure
from neurosurrogate.surrogate.parts.preprocessor.autoencoder import AEPreprocessor
from neurosurrogate.surrogate.parts.preprocessor.pca import PCAPreprocessor
from neurosurrogate.surrogate.runs import SurrogateRuns
from neurosurrogate.waveform.artifacts import (
    attractor_artifact,
    diff_artifact,
    simple_artifact,
)
from neurosurrogate.waveform.dynamics import (
    METRIC_KEYS,
    DynamicMetrics,
    extract_metric,
)

CONF_DIR = Path(__file__).resolve().parents[1] / "scripts" / "conf"
LATENT_DIMS = [1, 3]  # 単一 latent と複数 latent = 列構造 [V, z1..zN, u] の両端


@cache
def fit_surrogate(preset: str, n_components: int | None = None) -> Surrogate:
    """テスト専用プリセットを fit。テストの唯一の surrogate 生成口。
    n_components だけは preset 既定を上書きできる (列構造を振るテストのため)。
    fit は決定的 (pca 固定) なので同一引数は使い回す — テスト全体を 20s 以内に保つ。"""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"surrogate={preset}",
                *(
                    []
                    if n_components is None
                    else [f"surrogate.spec.n_components={n_components}"]
                ),
            ],
        )
    c = OmegaConf.to_container(cfg.surrogate, resolve=True)
    assert isinstance(c, dict)
    return Surrogate.fit(c)


def _train_comp(surrogate: Surrogate) -> int:
    """学習 comp の先頭 (代表)。既定では置換対象ノード全部で学習するので、
    単体モデルではこれが唯一の comp。"""
    return surrogate.spec.train_comp_ids()[0]


@pytest.fixture(scope="module")
def sindy() -> Surrogate:
    """代表 sindy surrogate。latent 次元に依らない性質のテストが共有する。"""
    return fit_surrogate("_test_hh_sindy")


def _spec_of(bundle: Surrogate) -> SimSpec:
    """学習データと同じ入力の評価仕様 (掃引軸なし = 点 1 つ)。学習側の指定も
    評価条件も同じ `SimSpec` なので、詰め替えずそのまま渡せる。"""
    return bundle.spec.dataset


def _simulate_view(series: EvalSeries, runs: SurrogateRuns) -> SeriesResults:
    """1 系列を run 軸に開いてその場で回す (保存を経由しない経路)。本番の描画入力は
    MLflow から読む `mlflow_io.report` の内部実装に相当するので、この経路はテストに
    だけ居る。"""
    return SeriesResults(
        run_column(series, None, None),
        tuple(
            run_column(series, rid, bundle)
            for rid, bundle in runs
            if bundle.spec.applicable(series)
        ),
    )


def _run_view(runs: SurrogateRuns, spec: SimSpec) -> SeriesResults:
    """spec をsurrogate run全部と原系で並走シミュした1系列。"""
    return _simulate_view(EvalSeries(spec=spec), runs)


@pytest.fixture(scope="module")
def sindy_view(sindy: Surrogate) -> SeriesResults:
    """単発 = 点 1 つ・run 1 本の系列。"""
    return _run_view(SurrogateRuns((("r0", sindy),)), _spec_of(sindy))


@pytest.fixture(scope="module")
def sindy_closure(sindy: Surrogate) -> SINDyBundle:
    """ξ / feature 式は SINDy 固有 (bundle.closure は表現非依存の Closure 型)。"""
    assert isinstance(sindy.closure, SINDyBundle)
    return sindy.closure


@pytest.mark.parametrize("n_components", LATENT_DIMS)
def test_sindy_replaced_sim_runs_at_any_latent_dim(n_components: int) -> None:
    """列構造 [V, z1..zN, u] は latent 次元によらず置換シミュまで通る。"""
    surrogate = fit_surrogate("_test_hh_sindy", n_components)
    assert isinstance(surrogate.closure, SINDyBundle)
    assert surrogate.closure.xi.shape[0] == n_components + 1  # V + latent
    assert len(surrogate.preprocessor.gate_inits) == n_components

    view = _run_view(SurrogateRuns((("r0", surrogate),)), _spec_of(surrogate))
    orig, surr = view.pair(0, view.column("r0"))
    v = access.potential(surr, _train_comp(surrogate))
    assert v.shape == access.time(orig).shape
    assert np.isfinite(v[0])


def test_sweep_metric_choices_are_all_extractable(sindy_view: SeriesResults) -> None:
    """UI が出す掃引 metric 選択肢は全て取り出せる = 選んだのに生成されないキーで
    黙って nan の図が出ることが無い (未知キーは extract_metric が KeyError)。"""
    orig, surr = sindy_view.pair(0, sindy_view.column("r0"))
    dm = DynamicMetrics(orig, surr, 0, sindy_view.series.spec.dt)
    assert all(extract_metric(dm, key)[1] is not None for key in METRIC_KEYS)
    with pytest.raises(KeyError):
        extract_metric(dm, "latency_error")


def test_sindy_draws_all_artifacts(sindy_view: SeriesResults, sindy: Surrogate) -> None:
    """1 セル (点 × run) の詳細図。潜在射影は callable で遅延評価される。"""
    orig, surr = sindy_view.pair(0, sindy_view.column("r0"))

    latent = transform_gate(sindy.preprocessor, orig, 0)
    artifacts = [
        diff_artifact(orig, latent, surr, 0),
        simple_artifact(orig),
        attractor_artifact(latent, surr, 0),
    ]
    assert [artifact.name for artifact in artifacts] == [
        "diff",
        "simple",
        "attractor",
    ]


def test_catalog_is_self_consistent() -> None:
    """カタログ (`scripts/catalog.py`) が自己整合: `SERIES` の全系列の電流が掃引点
    まで含めて構築でき、どの系列も comp 名を持つ (marimo の comp つまみは**選んだ
    1 系列**の適用先から選択肢を作るので、ここが空だとその系列を選ぶと何も選べない)。
    条件が型になった今、綴り間違いは import 時に落ちるので、ここで見るのは名前の
    対応だけ。単発系列も「点 1 つ」として同じ経路を通る。"""
    for series in SERIES.values():
        assert series.spec.net.names
        for spec in series.points:
            assert len(spec.current()) > 0
    # 点軸: 単発は点 1 つ、掃引は宣言した点数だけ
    assert len(SERIES["traub_soma_dc"].points) == 1
    assert len(SERIES["traub19_somastim"].points) == 5


def _sweep_series(values: list[float]) -> EvalSeries:
    """1 系列分の掃引宣言。"""
    return EvalSeries(
        spec=SimSpec(
            target="hh",
            current_type="lin&steady",
            dt=0.05,
            current_params={"duration": 30.0, "silence_duration": 0.0},
        ),
        param="value",
        values=values,
    )


def _sweep_view(runs: SurrogateRuns, values: list[float]) -> SeriesResults:
    """1 系列分の掃引をシミュした結果。"""
    return _simulate_view(_sweep_series(values), runs)


def test_trace_grid_rows_are_one_per_model(sindy: Surrogate) -> None:
    """波形格子の行 = 比べるモデル (run 軸)、列 = 点。1 レポートが並べるのは
    **1 系列の電流たち × N モデル**なので、行が増える軸は run だけ。"""
    runs = SurrogateRuns((("a", sindy), ("b", sindy)))
    view = _sweep_view(runs, [5.0, 10.0])
    artifact = traces_artifact(view, runs, "soma")
    assert isinstance(artifact.obj, Figure)
    assert len(artifact.obj.axes) == 2 * 2  # 2 モデル行 × 2 点列
    # 行がどの run かは行見出し (左列の y ラベル) で読む。
    assert [ax.get_ylabel() for ax in artifact.obj.axes] == ["a", "", "b", ""]


def test_series_view_columns_must_line_up_across_runs(
    sindy_view: SeriesResults,
) -> None:
    """列は自分の点数を、束は**列が同じ掃引を回したものか**を構築時に保証する
    (揃わない列を図の側で検出させない)。**由来の id は持たない** = 評価 run の同一性は
    ここに無く、描く中身だけの純粋なデータ (評価 run の id はレポート run の tag だけが
    持つ)。"""
    series = sindy_view.series
    with pytest.raises(ValueError, match="点数"):
        SeriesRun(series, "r0", [])
    # 別の掃引を回した列は束ねられない (点数が同じでも計算入力が違う)
    other = dc_replace(series, spec=dc_replace(series.spec, dt=series.spec.dt * 2))
    with pytest.raises(ValueError, match="記述の違う列"):
        SeriesResults(
            sindy_view.original,
            (SeriesRun(other, "r0", sindy_view.original_waves),),
        )
    # 原系の列に run_id は載らない / 置換系の run_id は欠けない
    with pytest.raises(ValueError, match="原系の列"):
        SeriesResults(SeriesRun(series, "r0", sindy_view.original_waves), ())
    with pytest.raises(ValueError, match="置換系の run_id"):
        SeriesResults(sindy_view.original, (sindy_view.original,))


def test_surrogate_runs_owns_run_axis(sindy: Surrogate) -> None:
    """surrogate列がrun名の一意性・選択順と結果軸との対応を保証する。"""
    runs = SurrogateRuns((("r0", sindy), ("r1", sindy)))
    assert runs.names == ("r0", "r1")
    assert runs.surrogate("r1") is sindy
    with pytest.raises(ValueError, match="重複"):
        SurrogateRuns((("r0", sindy), ("r0", sindy)))


@pytest.mark.parametrize("name", ("", ".", "..", "a/b", "a\\b", "a\0b"))
def test_surrogate_runs_rejects_names_unusable_as_path(
    name: str, sindy: Surrogate
) -> None:
    with pytest.raises(ValueError, match="pathに使えない"):
        SurrogateRuns(((name, sindy),))


def test_report_draws_the_results_at_hand_not_the_declaration(
    sindy_view: SeriesResults, sindy: Surrogate
) -> None:
    """描画は**手元の結果だけ**を見る (計算入力の設定と突き合わせない): 設定ファイル
    に宣言の無い結果 — 別セッションで回して artifact から読んだもの — もそのまま図に
    なる = 計算と描画が切れている。結果は系列名すら名乗らず (`SeriesResults` が持つのは
    点と run 軸だけ)、どの run に属するか (= 保存段) は**どの関数を呼んだか**で決まる
    (段を組むのは `scripts/mlflow_io`)。"""
    view = SeriesResults(sindy_view.original, sindy_view.surrs)
    assert [f.name for f in original_artifacts(view)] == ["current"]
    runs = SurrogateRuns((("r0", sindy),))
    # つまみは marimo の widget が作るのと同じ**全キー**。既定値は widget にしか無く、
    # ここは欠けたら KeyError で落ちる (握って別の値で描かない)。
    report_tuning = {"metric": "spike_count", "yauto": True, "ymin": 0.0, "ymax": 1.0}
    detail_tuning = {"detail_point": 0, "spike_orig": 0, "spike_surr": 0}
    waves = report_artifacts(view, runs, "soma", report_tuning)
    assert "traces" in {f.name for f in waves}
    detail = detail_artifacts(view, "r0", sindy, "soma", (), detail_tuning)
    assert {artifact.name for artifact in detail} == {
        "diff",
        "simple",
        "attractor",
        "metrics",
        "metrics_scalar",
    }
    # 手元の点数を超えた点 index は設定誤りとして落とす
    # (端へ丸めると指定と違う点の図が同じ保存名で出る)。
    for invalid_index in (-1, 99):
        with pytest.raises(ValueError, match="点 index"):
            detail_artifacts(
                view,
                "r0",
                sindy,
                "soma",
                (),
                detail_tuning | {"detail_point": invalid_index},
            )
    # 適用先に無い comp は名前解決がそのまま KeyError (先回りして検証しない)
    with pytest.raises(KeyError):
        detail_artifacts(view, "r0", sindy, "nope", (), detail_tuning)
    # つまみのキーが欠けていれば、既定値で埋めずに KeyError
    with pytest.raises(KeyError, match="detail_point"):
        detail_artifacts(view, "r0", sindy, "soma", (), {})


def test_surrogate_artifacts_come_from_the_run_itself_not_a_declaration(
    sindy: Surrogate,
) -> None:
    """モデル側の図は**その run が自分について描けるもの**で決まる (何を描くかの
    宣言を受け取らない)。描く対象は学習 run そのものなので、レポートを増やしても
    同じ図が複製されない。run 横断のサマリ表はここに無い (選択した N 本の産物)。"""
    artifacts = surrogate_artifacts(sindy, ())
    # SINDy = ξ heatmap を持つ表現なので model 図が出る
    assert "model" in {artifact.name for artifact in artifacts}
    assert not any("summary" in artifact.name for artifact in artifacts)
    assert summary_artifact(SurrogateRuns((("r0", sindy),))).name == "summary"


def test_artifact_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
    sindy: Surrogate,
) -> None:
    """描画失敗を正常な成果物へ変換せず、呼び出し元へ伝播させる。"""

    def boom() -> Figure:
        raise KeyError("missing var")

    monkeypatch.setattr(artifact_bundle, "train_raw_artifact", lambda *_: boom())
    with pytest.raises(KeyError, match="missing var"):
        surrogate_artifacts(sindy)


def test_view_comps_limit_drawn_traces(
    sindy_view: SeriesResults, sindy: Surrogate
) -> None:
    """表示 comp 制限 (UI の view_comps) が全 comp を並べる図に効く: 対象外だけを
    指定するとパネル/trace が消える。

    **学習データ図にも効く** — 以前は `comps` が署名にあるだけで body が soma 固定
    (学会前のその場しのぎ) で、UI の選択が黙って無視されていた。"""
    ds = sindy_view.original_waves[0]
    limited, full = simple_artifact(ds, comps=[]).obj, simple_artifact(ds).obj
    assert isinstance(limited, Figure) and isinstance(full, Figure)
    assert len(limited.axes) < len(full.axes)

    def n_traces(fig: object) -> int:
        assert isinstance(fig, Figure)
        return sum(len(ax.lines) for ax in fig.axes)

    # 学習 comp を 1 つも含まない選択では comp 由来の trace が消える。
    # train_raw は先頭 comp が要るゲート段ごと落ち、train_preprocessed は段は
    # 潜在次元で決まるので段内の trace だけが消える → 数え方は trace で揃える。
    for artifact in (train_raw_artifact, train_preprocessed_artifact):
        assert n_traces(artifact(sindy, comps=[]).obj) < n_traces(artifact(sindy).obj)
    assert len(train_raw_artifact(sindy, comps=[]).obj.axes) == 2  # type: ignore[union-attr]

    # 学習 comp を名指しすれば全部描いたときと一致する (単体 hh は学習 comp 1 つ)
    picked = train_raw_artifact(sindy, comps=[_train_comp(sindy)])
    assert n_traces(picked.obj) == n_traces(train_raw_artifact(sindy).obj)


def test_pca_metrics_report_per_component_and_cumulative_ratio(
    sindy: Surrogate,
) -> None:
    """PCA metrics は保持成分ごとの寄与率 (連番) と累積を出す。累積は保持成分の
    ratio 和と一致し 0〜1、full スペクトルは保持成分以上の長さを持つ。"""
    prep = sindy.preprocessor
    assert isinstance(prep, PCAPreprocessor)
    m = prep.metrics()
    n_kept = len(prep.explained_variance_ratio)
    assert [f"pca/explained_variance_ratio_{i + 1}" for i in range(n_kept)] == [
        k for k in m if k.startswith("pca/explained_variance_ratio_")
    ]
    cumulative = m["pca/cumulative_explained_variance_ratio"]
    assert cumulative == pytest.approx(float(prep.explained_variance_ratio.sum()))
    assert 0.0 <= cumulative <= 1.0 + 1e-9
    assert len(prep.full_explained_variance_ratio) >= n_kept


def test_preprocessor_artifact_scree_for_pca_none_for_ae(
    sindy: Surrogate,
) -> None:
    """PCA は scree 成果物を返し、AE は固有成果物なし → None。"""
    artifact = preprocessor_artifact(sindy.preprocessor)
    assert artifact is not None and artifact.name == "pca_scree"
    ae = fit_surrogate("_test_traub_hybrid")  # preprocessor_type=ae ではないので確認
    if isinstance(ae.preprocessor, AEPreprocessor):
        assert preprocessor_artifact(ae.preprocessor) is None


def test_train_artifacts_render_from_reloaded_surrogate(
    sindy: Surrogate, tmp_path: Path
) -> None:
    """学習データ図は save/load を跨いで描ける: 軌道は保存されず spec +
    scope / ansatz の規則から再生成される (marimo が run ロード毎に描く経路)。"""
    sindy.save(tmp_path)
    reloaded = Surrogate.load(tmp_path)
    assert reloaded.spec == sindy.spec  # spec は JSON で round-trip
    assert reloaded.spec.train_comp_ids() == [_train_comp(sindy)]
    assert reloaded.n_training_gates == len(sindy.spec.comp_type.gate_names)

    names = [
        artifact.name
        for artifact in (
            train_raw_artifact(reloaded),
            train_preprocessed_artifact(reloaded),
            train_recon_artifact(reloaded),
            train_v_coverage_artifact(reloaded),
            train_manifold_artifact(reloaded),
        )
    ]
    assert names == [
        "train_raw",
        "train_preprocessed",
        "train_recon",
        "train_v_coverage",
        "train_manifold",
    ]


def test_train_inputs_match_identified_columns(sindy: Surrogate) -> None:
    """train_preprocessed 図が描くのは fit が同定器へ渡したものと同一 — 列名/軌道数が
    閉包項の列構造と一致することで担保する (view は fit と同じ関数を呼ぶ)。"""
    inputs = sindy.training_inputs()
    assert isinstance(sindy.closure, SINDyBundle)
    assert [str(s) for s in sindy.closure.columns] == inputs.x_names + inputs.u_names
    assert [x.shape[1] for x in inputs.x] == [len(inputs.x_names)] * len(inputs.x)
    # 軌道数は scope の選択規則と一致 — 出所 comp は片方だけが持つ
    assert len(inputs.u) == len(inputs.x) == len(sindy.spec.train_comp_ids())


def test_hybrid_training_scope_covers_all_replaceable_comps() -> None:
    """hybrid は置換対象 comp 全部の軌道で学習し、
    学習ゲートは physics 分離後の先頭 n_learned 本に限られる。"""
    surrogate = fit_surrogate("_test_traub_hybrid")
    assert surrogate.spec.train_comp_ids() == [
        i
        for i, comp in enumerate(surrogate.spec.dataset.net.nodes)
        if comp.type == surrogate.spec.comp_type
    ]
    # physics 側へ回した状態 (Ca サブ系) は学習に含めない
    n_gate = surrogate.n_training_gates
    assert n_gate == len(surrogate.spec.comp_type.gate_names) - len(
        TRAUB_EXTRA_GATE_NAMES
    )
    assert surrogate.training_gates()[0].shape[1] == n_gate


def test_ae_preprocessor_path_runs() -> None:
    """AE 経路の smoke (pca 固定の他テストが踏まない encode/decode を通す)。
    epochs を切り詰めるので再構成品質は問わない — 形状と潜在次元の整合のみ。"""
    surrogate = fit_surrogate("_test_hh_ae")
    gate = surrogate.training_gates()[0]
    latent = surrogate.preprocessor.encode(gate)
    assert latent.shape == (gate.shape[0], 2)
    assert np.asarray(surrogate.preprocessor.decode(jnp.asarray(latent))).shape == (
        gate.shape
    )
    assert len(surrogate.preprocessor.gate_inits) == 2


def test_feature_exprs_align_with_xi_columns(sindy_closure: SINDyBundle) -> None:
    """feature 式列は xi の列と 1:1 (fit が pysindy 名との一致を検証済み)。"""
    assert len(sindy_closure.feature_exprs) == sindy_closure.xi.shape[1]


def test_duplicate_library_types_are_rejected(sindy_closure: SINDyBundle) -> None:
    """library type は互いに素 → 同 type 2 回で feature 式が重複しエラー。"""
    library = FeatureLibrary.build(
        sindy_closure.library_specs + sindy_closure.library_specs, sindy_closure.roles
    )
    with pytest.raises(ValueError, match="feature 重複"):
        library.bound_exprs(sindy_closure.columns)


def test_feature_tex_drops_model_suffix(sindy_closure: SINDyBundle) -> None:
    """レート関数は未定義 Function → sympy が自動でギリシャ文字化。model は下付きへ
    回すが、heatmap の軸ラベルでは冗長 → 落とす (tex は括弧へ整形して残す)。"""
    texs = [feature_tex(e) for e in sindy_closure.feature_exprs]
    assert all(t.startswith("$") and t.endswith("$") for t in texs)
    assert any(r"\alpha_{m}{\left(V \right)}" in t for t in texs)
    assert not any("(hh)" in t for t in texs)
    full = [tex(e) for e in sindy_closure.feature_exprs]
    assert any(r"\alpha_{m(hh)}{\left(V \right)}" in t for t in full)


@pytest.mark.parametrize(
    ("preset", "extra_names"),
    [
        ("_test_traub_hybrid", TRAUB_EXTRA_GATE_NAMES),
        ("_test_traub_hybrid_sr_physics", TRAUB_SR_EXTRA_GATE_NAMES),
    ],
)
def test_hybrid_traub_transplants_across_heterogeneous_compartments(
    preset: str, extra_names: list[str]
) -> None:
    """hybrid traub は Ca サブ系を physics へ分離し純電位依存ゲートのみ学習 →
    Ca params (phi_area/g_Ca) がノード毎に違う traub19 全 comp を 1 サロゲートで置換
    できる (compatible=True)。surr state は latent + physics 状態で、分割位置は
    preset (spec.physics_type) が決める → 両分割で kernel が有限に走ることを確認。
    preprocessor は AE の乱数初期化で fit 品質 (=有限性) がブレる → 決定的な pca に固定
    (主眼は physics 積分経路であり AE 再構成品質ではない)。"""
    surrogate = fit_surrogate(preset)
    assert surrogate.surr_comp_type.gate_names[-len(extra_names) :] == extra_names

    traub19 = SimSpec(target="traub19", current_type="train", dt=0.01)
    # phi_area/g_Ca が異なる 19 comp すべてが置換対象。pre-B は soma のみ一致で
    # ValueError だった (Ca params が latent に焼込まれ params 一致必須だったため)。
    assert surrogate.spec.replacement_targets(traub19.net) == set(traub19.net.names)

    # 置換シミュ (XI/Q を各ノード params で physics 積分) が有限に走る。
    v = access.potential(
        unified_simulator(surrogate.apply(surrogate.spec.dataset.materialize())),
        _train_comp(surrogate),
    )
    assert np.isfinite(v).all()


def test_traub19_soma_model_replaces_only_soma() -> None:
    """適用先モデル traub19_soma は soma だけ traub 型に残し dendrite をダミー型に
    する → comp_type=traub の学習を **preset 変更なし** で soma 1 ノードだけへ適用
    できる (置換範囲を絞る新軸を spec へ足さず、適用先モデル側で絞る)。dendrite 18 個
    は置換対象外のまま残り、置換シミュが有限に走る。"""
    surrogate = fit_surrogate("_test_traub_hybrid")  # comp_type=traub, 単体 traub 教師
    ds = SimSpec(
        target="traub19_soma",
        current_type="train",
        dt=0.01,
        current_params={"duration": 180},  # smoke: 配線確認のみ (本番は長時間)
    )
    assert surrogate.spec.replacement_targets(ds.net) == {"soma"}
    # soma だけ traub 型、dendrite はダミー型 traub_ (置換対象外)
    assert {n.name for n in ds.net.nodes if n.type.name == "traub"} == {"soma"}

    v = access.potential(
        unified_simulator(surrogate.apply(ds.materialize())),
        ds.net.name_to_idx("soma"),
    )
    assert np.isfinite(v).all()


def test_traub19_soma_dendstim_injects_into_dendrite() -> None:
    """dend 刺激版も soma だけ置換対象 (traub 型 = soma のみ) だが、電流注入先は
    dendrite。刺激点が soma でないこと + 置換シミュが有限に走ることを確認。"""
    surrogate = fit_surrogate("_test_traub_hybrid")
    ds = SimSpec(
        target="traub19_soma_dendstim",
        current_type="train",
        dt=0.01,
        current_params={"duration": 180},
    )
    assert ds.net.stim != "soma"  # 注入先は dendrite
    assert surrogate.spec.replacement_targets(ds.net) == {"soma"}

    v = access.potential(
        unified_simulator(surrogate.apply(ds.materialize())),
        ds.net.name_to_idx("soma"),
    )
    assert np.isfinite(v).all()


def test_ude_joint_fit_updates_the_preprocessor() -> None:
    """UDE の要: 潜在座標が「先に固定される前処理」でなくなること。

    setup は AE を再構成 MSE で fit してから ansatz.fit を呼ぶ (hybrid と同じ順) が、
    UDE はその encoder/decoder を初期値として受け取り、軌道ロスで更新して書き戻す。
    AE 単体の fit は乱数種固定で決定的 → 同じ入力で再現したものと重みが違えば、
    joint 学習が実際に座標を動かしたことになる。
    """
    surrogate = fit_surrogate("_test_traub_ude")
    assert isinstance(surrogate.closure, UDEClosure)
    assert isinstance(surrogate.preprocessor, AEPreprocessor)  # ude は ae 固定
    gate = np.concatenate(surrogate.training_gates(), axis=0)
    ae_only = AEPreprocessor.fit(gate, surrogate.spec.n_components, {"epochs": 20})

    assert not np.allclose(
        surrogate.preprocessor.dec_params["W2"], ae_only.dec_params["W2"]
    )
    # 書き戻し後に再計算される派生物 (kernel の初期潜在・再構成指標) も joint 後の値
    assert len(surrogate.preprocessor.gate_inits) == surrogate.spec.n_components
    assert surrogate.preprocessor.reconstruction_mse != ae_only.reconstruction_mse


def test_ude_traub_transplants_across_heterogeneous_compartments() -> None:
    """UDE も hybrid と同じ kernel 骨格に載る (差分は潜在方程式の表現だけ) →
    Ca サブ系の physics 分離と traub19 全 comp への移植がそのまま成り立つ。"""
    surrogate = fit_surrogate("_test_traub_ude")
    assert surrogate.surr_comp_type.gate_names[-len(TRAUB_EXTRA_GATE_NAMES) :] == (
        TRAUB_EXTRA_GATE_NAMES
    )
    traub19 = SimSpec(target="traub19", current_type="train", dt=0.01)
    assert surrogate.spec.replacement_targets(traub19.net) == set(traub19.net.names)

    v = access.potential(
        unified_simulator(surrogate.apply(surrogate.spec.dataset.materialize())),
        _train_comp(surrogate),
    )
    assert np.isfinite(v).all()


def test_ude_rejects_non_learnable_preprocessor() -> None:
    """UDE は encoder/decoder を学習変数として更新する → 更新できない表現 (PCA) を
    渡されたら黙って前処理固定に退化せず、その場で落ちる。"""
    pca_based = fit_surrogate("_test_traub_hybrid")  # preprocessor_type=pca の preset
    with pytest.raises(ValueError, match="preprocessor_type=ae"):
        UDEAnsatz().fit(
            pca_based.spec,
            pca_based.training_data,
            pca_based.preprocessor,
            {"epochs": 1},
        )


class _IdentityPreprocessor(Preprocessor):
    """恒等 preprocessor: latent==gate を無演算 passthrough する検証専用実装。

    真の式を潜在方程式へ入れたとき置換 kernel が原系と bit 一致することを担保する
    ため、decode/encode を一切の演算なしで素通しする (fit 経路は持たない手組み)。
    """

    def __init__(self, gate_inits: list[float]) -> None:
        self.gate_inits = gate_inits
        self.reconstruction_mse = 0.0
        self.reconstruction_mse_ratio = 0.0

    @classmethod
    def fit(
        cls, train_gate: np.ndarray, n_components: int, spec: dict
    ) -> "Preprocessor":
        raise NotImplementedError("恒等 preprocessor は手組み専用")

    def encode(self, x: np.ndarray) -> np.ndarray:
        return x

    def decode(self, state: jnp.ndarray) -> jnp.ndarray:
        return state

    def metrics(self) -> dict:
        return {}

    def opcost(self) -> OpCost:
        return OpCost()

    @property
    def n_features(self) -> int:
        return len(self.gate_inits)


class _NullClosure(Closure):
    """潜在方程式は自由関数 (dlatent_fn) 側で与える → opcost/metrics だけの殻。"""

    def metrics(self) -> dict[str, float]:
        return {}

    def opcost(self) -> OpCost:
        return OpCost()


def test_original_dynamics_injected_reproduces_potential_exactly() -> None:
    """恒等サロゲート: 前処理=恒等 (latent==gate)・潜在方程式=真の HH ゲート式 を
    hybrid kernel へ差し込むと、置換系の電位遷移が原系と **完全一致** する。

    サロゲート積分経路 (座標変換・state レイアウト・physics dV/dt・euler・型差替) に
    近似以外の齟齬が無いことの担保 — 真の右辺を入れれば差はゼロでなければならない。
    """
    surrogate = fit_surrogate("_test_hh_hybrid")  # comp_type=hh, n_components=3
    spec = surrogate.spec
    p = spec.train_comp().resolved_params
    assert isinstance(p, HHParams)

    # 潜在方程式 = 原 HH の dgate そのまま (latent==gate なので引数もそのまま渡せる)。
    def exact_dlatent(latent: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        v_rel = v - p.E_REST
        return jnp.stack(
            [dmdt(v_rel, latent[0]), dhdt(v_rel, latent[1]), dndt(v_rel, latent[2])]
        )

    class ExactHybridAnsatz(HybridAnsatz[_NullClosure]):
        def fit(self, spec, training_data, preprocessor, config):
            raise NotImplementedError

        def dlatent(self, spec, preprocessor, closure):
            return exact_dlatent

    surr_type = ExactHybridAnsatz().surr_comp_type(
        spec,
        _IdentityPreprocessor(hh_inits(p)[1:]),  # 初期ゲート = 原 init のゲート部
        _NullClosure(),
    )
    replaced = dc_replace(
        spec.dataset.materialize(),
        net=dc_replace(
            spec.dataset.net,
            nodes=[
                dc_replace(node, type=surr_type)
                if node.type == spec.comp_type
                else node
                for node in spec.dataset.net.nodes
            ],
        ),
    )

    comp = _train_comp(surrogate)
    assert np.array_equal(
        access.potential(unified_simulator(spec.dataset.materialize()), comp),
        access.potential(unified_simulator(replaced), comp),
    )


def test_hybrid_opcost_includes_decode() -> None:
    """hybrid の kernel は毎ステップ decode を呼ぶ → OpCost に計上されている。"""
    surrogate = fit_surrogate("_test_hh_hybrid")
    assert isinstance(surrogate.closure, SINDyBundle)  # opcost は表現固有
    decode_cost = surrogate.preprocessor.opcost()
    # PCA decode: gate ごとに latent 数の積 + 同数の加減 (3 latent x 3 gate)
    assert (decode_cost.mul, decode_cost.pm) == (9, 9)
    assert surrogate.surr_comp_type.opcost == (
        decode_cost
        + HYBRID_PHYSICS[
            surrogate.spec.physics_type or surrogate.spec.comp_type.name
        ].dv_cost
        + surrogate.closure.opcost()
    )
