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
from catalog import REPORT, SERIES
from hydra import compose, initialize_config_dir
from matplotlib.figure import Figure
from omegaconf import OmegaConf

from neurosurrogate.core import access
from neurosurrogate.core.opcost import OpCost
from neurosurrogate.core.simulator import unified_simulator
from neurosurrogate.neurons.compartments.hh import HHParams, dhdt, dmdt, dndt, hh_inits
from neurosurrogate.neurons.compartments.traub import (
    TRAUB_EXTRA_GATE_NAMES,
    TRAUB_SR_EXTRA_GATE_NAMES,
)
from neurosurrogate.plotting import collect, new_figure
from neurosurrogate.report.build import eval_entries, model_entries
from neurosurrogate.report.grid import trace_grid_fig
from neurosurrogate.report.results import SeriesView, simulate_views
from neurosurrogate.report.spec import Report, Tuning
from neurosurrogate.sim.eval import EvalSeries
from neurosurrogate.sim.spec import SimSpec
from neurosurrogate.surrogate.ansatz.impl.hybrid import HybridAnsatz
from neurosurrogate.surrogate.ansatz.impl.hybrid_kernel import (
    hybrid_physics,
    hybrid_surr_comp_type,
)
from neurosurrogate.surrogate.ansatz.impl.ude import UDEAnsatz
from neurosurrogate.surrogate.bundle import SurrogateBundle
from neurosurrogate.surrogate.closure.base import Closure
from neurosurrogate.surrogate.closure.sindy import SINDyBundle
from neurosurrogate.surrogate.closure.sindy.entry import FeatureLibrary
from neurosurrogate.surrogate.closure.ude import UDEClosure
from neurosurrogate.surrogate.diagnostics import preprocessed_latent
from neurosurrogate.surrogate.figures import preprocessor_figs, train_figs
from neurosurrogate.surrogate.figures.model import feature_tex, tex
from neurosurrogate.surrogate.preprocessor.base import Preprocessor
from neurosurrogate.surrogate.preprocessor.impl.autoencoder import AEPreprocessor
from neurosurrogate.surrogate.preprocessor.impl.pca import PCAPreprocessor
from neurosurrogate.surrogate.replace import (
    apply_surrogate,
    replace_nodes,
    replaceables,
)
from neurosurrogate.waveform import cell_figs, panels_simple
from neurosurrogate.waveform.dynamics import (
    METRIC_KEYS,
    DynamicMetrics,
    extract_metric,
)

CONF_DIR = Path(__file__).resolve().parents[1] / "scripts" / "conf"
LATENT_DIMS = [1, 3]  # 単一 latent と複数 latent = 列構造 [V, z1..zN, u] の両端


@cache
def fit_surrogate(preset: str, n_components: int | None = None) -> SurrogateBundle:
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
                    else [f"surrogate.meta.n_components={n_components}"]
                ),
            ],
        )
    c = OmegaConf.to_container(cfg.surrogate, resolve=True)
    assert isinstance(c, dict)
    return SurrogateBundle.setup(c)


def _train_comp(surrogate: SurrogateBundle) -> int:
    """学習 comp の先頭 (代表)。既定では置換対象ノード全部で学習するので、
    単体モデルではこれが唯一の comp。"""
    return surrogate.ansatz.train_source(surrogate.meta).comp_ids[0]


@pytest.fixture(scope="module")
def sindy() -> SurrogateBundle:
    """代表 sindy surrogate。latent 次元に依らない性質のテストが共有する。"""
    return fit_surrogate("_test_hh_sindy")


def _spec_of(bundle: SurrogateBundle) -> SimSpec:
    """学習データと同じ入力の評価仕様 (掃引軸なし = 点 1 つ)。学習側の指定も
    評価条件も同じ `SimSpec` なので、詰め替えずそのまま渡せる。"""
    return bundle.meta.dataset


def _run_view(
    bundles: dict[str, SurrogateBundle], spec: SimSpec, series: str = "hh_dc"
) -> SeriesView:
    """spec を bundles (run_id → surrogate) 全部と原系で並走シミュした 1 系列。"""
    return simulate_views({series: EvalSeries(spec=spec)}, bundles)[series]


@pytest.fixture(scope="module")
def sindy_view(sindy: SurrogateBundle) -> SeriesView:
    """単発 = 点 1 つ・run 1 本の系列。"""
    return _run_view({"r0": sindy}, _spec_of(sindy))


@pytest.fixture(scope="module")
def sindy_closure(sindy: SurrogateBundle) -> SINDyBundle:
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

    orig, surr = _run_view({"r0": surrogate}, _spec_of(surrogate)).pair(0, "r0")
    v = access.potential(surr.dataset, _train_comp(surrogate))
    assert v.shape == access.time(orig.dataset).shape
    assert np.isfinite(v[0])


def test_sweep_metric_choices_are_all_extractable(sindy_view: SeriesView) -> None:
    """UI が出す掃引 metric 選択肢は全て取り出せる = 選んだのに生成されないキーで
    黙って nan の図が出ることが無い (未知キーは extract_metric が KeyError)。"""
    orig, surr = sindy_view.pair(0, "r0")
    dm = DynamicMetrics(orig.dataset, surr.dataset, 0, surr.spec.dt)
    assert all(extract_metric(dm, key)[1] is not None for key in METRIC_KEYS)
    with pytest.raises(KeyError):
        extract_metric(dm, "latency_error")


def test_sindy_draws_all_figs(sindy_view: SeriesView, sindy: SurrogateBundle) -> None:
    """1 セル (点 × run) の詳細図。潜在射影は callable で遅延評価される。"""
    orig, surr = sindy_view.pair(0, "r0")
    figs = cell_figs(
        orig.dataset,
        surr.dataset,
        0,
        lambda: preprocessed_latent(sindy, orig.spec.net, orig.dataset, 0),
    )
    assert [name for name, _ in figs] == ["diff", "simple", "attractor"]


def test_catalog_is_self_consistent() -> None:
    """カタログ (`scripts/catalog.py`) が自己整合: `SERIES` の全系列の電流が掃引点
    まで含めて構築でき、`REPORT` は `SERIES` と**同じキー空間**を張る (1 系列 =
    1 レポート → 描き方の無い系列も、回さない系列の描き方も存在しない)。条件も宣言も
    型になった今、綴り間違いは import 時に落ちるので、ここで見るのは**名前の対応**
    だけ。単発系列も「点 1 つ」として同じ経路を通る。"""
    for series in SERIES.values():
        for spec in series.points:
            assert len(spec.current()) > 0
    # 点軸: 単発は点 1 つ、掃引は宣言した点数だけ
    assert len(SERIES["traub_soma_dc"].points) == 1
    assert len(SERIES["traub19_somastim"].points) == 5

    assert set(REPORT) == set(SERIES)


def _sweep_specs(name: str, values: list[float]) -> dict[str, EvalSeries]:
    """1 系列分のカタログ項目。"""
    return {
        name: EvalSeries(
            spec=SimSpec(
                target="hh",
                current_type="lin&steady",
                dt=0.05,
                current_params={"duration": 30.0, "silence_duration": 0.0},
            ),
            param="value",
            values=values,
        )
    }


def _sweep_view(
    bundles: dict[str, SurrogateBundle], name: str, values: list[float]
) -> SeriesView:
    """1 系列分の掃引をシミュした結果。"""
    return simulate_views(_sweep_specs(name, values), bundles)[name]


def test_trace_grid_rows_are_one_per_model(sindy: SurrogateBundle) -> None:
    """波形格子の行 = 比べるモデル (run 軸)、列 = 点。1 レポートが並べるのは
    **1 系列の電流たち × N モデル**なので、行が増える軸は run だけ。"""
    bundles = {"r0": sindy, "r1": sindy}
    names = {"r0": "a", "r1": "b"}
    view = _sweep_view(bundles, "a", [5.0, 10.0])
    fig = trace_grid_fig(view, names, "soma")
    assert len(fig.axes) == 2 * 2  # 2 モデル行 × 2 点列
    # 行名は描かない (格子は列見出しと凡例で読む)。
    assert not any(ax.get_ylabel() for ax in fig.axes)


def test_series_view_columns_must_line_up_across_runs(sindy_view: SeriesView) -> None:
    """`SeriesView` は点軸の列が run 間で揃っていることを構築時に保証する
    (揃わない列を図の側で検出させない)。"""
    with pytest.raises(ValueError, match="点数"):
        SeriesView("s", sindy_view.points, {"r0": []})


def test_report_draws_the_results_at_hand_not_the_declaration(
    sindy_view: SeriesView, sindy: SurrogateBundle
) -> None:
    """描画は**手元の結果だけ**を見る (計算入力の設定と突き合わせない): 設定ファイル
    に宣言の無い系列名 — 別セッションで回して artifact から読んだ結果 — もそのまま
    図になる = 計算と描画が切れている。系列名は `dest` 側の関心なので成果物の名前に
    出ない (1 レポート = 1 系列 → 名前に系列名を混ぜる必要がない)。"""
    renamed = dc_replace(sindy_view, name="読んだ系列")
    entries = eval_entries(renamed, {"r0": sindy}, Report(eval_comp="soma"))
    assert {"current", "traces"} <= {e.name for e in entries}
    assert not any(e.name.startswith("読んだ系列") for e in entries)
    # つまみ (`Tuning`) はカタログでなく描画時の引数。詳細図は点 index を名前に
    # 持つので、つまみを動かしても前の点を上書きしない。
    assert any("/p0/" in e.name for e in entries)
    moved = eval_entries(
        renamed, {"r0": sindy}, Report(eval_comp="soma"), Tuning(detail_point=99)
    )
    # 手元の点数へ丸める (設定が実際の点数を超えていても描く)
    last = len(renamed.points) - 1
    assert any(f"/p{last}/" in e.name for e in moved)


def test_model_figs_come_from_the_run_itself_not_a_declaration(
    sindy: SurrogateBundle,
) -> None:
    """モデル側の図は**その run が自分について描けるもの**で決まる (何を描くかの
    宣言を受け取らない)。比べる N 本すべてを描く = 「代表 1 本だけ」の恣意が無い。"""
    entries = model_entries({"r0": sindy, "r1": sindy})
    names = [e.name for e in entries]
    assert "summary" in names  # run 横断の学習側サマリ表は 1 枚
    # SINDy = ξ heatmap を持つ表現なので model 図が出る。それが run ごとに揃う。
    per_run = {name.split("/")[0] for name in names if "/" in name}
    assert len(per_run) == 2
    assert all(f"{run}/model" in names for run in per_run)


def test_failed_figs_fold_into_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """描画 job の失敗は列を保ったまま error 図へ畳む = 1 図の失敗で他の図まで
    落とさない。"""

    def boom() -> Figure:
        raise KeyError("missing var")

    assert [name for name, _ in collect({"ok": new_figure, "ng": boom})] == ["ok", "ng"]
    assert "ng" in capsys.readouterr().err  # 失敗は握り潰さず stderr へも出す


def test_view_comps_limit_drawn_traces(
    sindy_view: SeriesView, sindy: SurrogateBundle
) -> None:
    """表示 comp 制限 (UI の view_comps) が全 comp を並べる図に効く: 対象外だけを
    指定するとパネル/trace が消え、学習 comp を指定した学習データ図は描ける。"""
    ds = sindy_view.points[0].dataset
    assert len(panels_simple(ds, comps=[])) < len(panels_simple(ds))
    assert [name for name, _ in train_figs(sindy, comps=[_train_comp(sindy)])] == [
        name for name, _ in train_figs(sindy)
    ]


def test_pca_metrics_report_per_component_and_cumulative_ratio(
    sindy: SurrogateBundle,
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


def test_preprocessor_figs_scree_for_pca_empty_for_ae(
    sindy: SurrogateBundle,
) -> None:
    """PCA は scree 図を返し、AE は固有図なし → 空列 (closure_figs と同型)。"""
    assert [name for name, _ in preprocessor_figs(sindy.preprocessor)] == ["pca_scree"]
    ae = fit_surrogate("_test_traub_hybrid")  # preprocessor_type=ae ではないので確認
    if isinstance(ae.preprocessor, AEPreprocessor):
        assert preprocessor_figs(ae.preprocessor) == []


def test_train_figs_render_from_reloaded_surrogate(
    sindy: SurrogateBundle, tmp_path: Path
) -> None:
    """学習データ図は save/load を跨いで描ける: 軌道は保存されず meta +
    ansatz.train_source から再生成される (marimo が run ロード毎に描く経路)。"""
    sindy.save(tmp_path)
    reloaded = SurrogateBundle.load(tmp_path)
    assert reloaded.meta == sindy.meta  # meta は JSON で round-trip
    source = reloaded.ansatz.train_source(reloaded.meta)
    assert source.comp_ids == [_train_comp(sindy)]  # 単体 hh モデル → 1 comp
    assert source.n_gate == len(sindy.meta.comp_type.gate_names)  # 全ゲート

    names = [name for name, _ in train_figs(reloaded)]
    assert names == [
        "train_raw",
        "train_preprocessed",
        "train_recon",
        "train_v_coverage",
        "train_manifold",
    ]


def test_train_inputs_match_identified_columns(sindy: SurrogateBundle) -> None:
    """train_preprocessed 図が描くのは fit が同定器へ渡したものと同一 — 列名/軌道数が
    閉包項の列構造と一致することで担保する (view は fit と同じ関数を呼ぶ)。"""
    inputs = sindy.ansatz.train_inputs(sindy.meta, sindy.train_xr, sindy.preprocessor)
    assert isinstance(sindy.closure, SINDyBundle)
    assert [str(s) for s in sindy.closure.columns] == inputs.x_names + inputs.u_names
    assert [x.shape[1] for x in inputs.x] == [len(inputs.x_names)] * len(inputs.x)
    # 軌道数は選択規則 (TrainSource.comp_ids) と一致 — 出所 comp は片方だけが持つ
    assert (
        len(inputs.u)
        == len(inputs.x)
        == len(sindy.ansatz.train_source(sindy.meta).comp_ids)
    )


def test_hybrid_train_source_covers_all_replaceable_comps() -> None:
    """hybrid は置換対象 comp 全部の軌道で学習 → train_source がそれを記録し、
    学習ゲートは physics 分離後の先頭 n_learned 本に限られる。"""
    surrogate = fit_surrogate("_test_traub_hybrid")
    source = surrogate.ansatz.train_source(surrogate.meta)
    assert source.comp_ids == [
        i
        for i, comp in enumerate(surrogate.meta.dataset.net.nodes)
        if comp.type == surrogate.meta.comp_type
    ]
    # physics 側へ回した状態 (Ca サブ系) は学習に含めない
    assert source.n_gate == len(surrogate.meta.comp_type.gate_names) - len(
        TRAUB_EXTRA_GATE_NAMES
    )
    assert source.stacked_gate(surrogate.train_xr).shape[1] == source.n_gate


def test_ae_preprocessor_path_runs() -> None:
    """AE 経路の smoke (pca 固定の他テストが踏まない encode/decode を通す)。
    epochs を切り詰めるので再構成品質は問わない — 形状と潜在次元の整合のみ。"""
    surrogate = fit_surrogate("_test_hh_ae")
    source = surrogate.ansatz.train_source(surrogate.meta)
    gate = source.gate(surrogate.train_xr, _train_comp(surrogate))
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
    preset (meta.physics_type) が決める → 両分割で kernel が有限に走ることを確認。
    preprocessor は AE の乱数初期化で fit 品質 (=有限性) がブレる → 決定的な pca に固定
    (主眼は physics 積分経路であり AE 再構成品質ではない)。"""
    surrogate = fit_surrogate(preset)
    assert surrogate.surr_comp_type.gate_names[-len(extra_names) :] == extra_names

    traub19 = SimSpec(target="traub19", current_type="train", dt=0.01)
    # phi_area/g_Ca が異なる 19 comp すべてが置換対象。pre-B は soma のみ一致で
    # ValueError だった (Ca params が latent に焼込まれ params 一致必須だったため)。
    assert replaceables(surrogate.meta, traub19.net) == set(traub19.net.names)

    # 置換シミュ (XI/Q を各ノード params で physics 積分) が有限に走る。
    v = access.potential(
        unified_simulator(
            apply_surrogate(surrogate, surrogate.meta.dataset.materialize())
        ),
        _train_comp(surrogate),
    )
    assert np.isfinite(v).all()


def test_traub19_soma_model_replaces_only_soma() -> None:
    """適用先モデル traub19_soma は soma だけ traub 型に残し dendrite をダミー型に
    する → comp_type=traub の学習を **preset 変更なし** で soma 1 ノードだけへ適用
    できる (置換範囲を絞る新軸を meta へ足さず、適用先モデル側で絞る)。dendrite 18 個
    は置換対象外のまま残り、置換シミュが有限に走る。"""
    surrogate = fit_surrogate("_test_traub_hybrid")  # comp_type=traub, 単体 traub 教師
    ds = SimSpec(
        target="traub19_soma",
        current_type="train",
        dt=0.01,
        current_params={"duration": 180},  # smoke: 配線確認のみ (本番は長時間)
    )
    assert replaceables(surrogate.meta, ds.net) == {"soma"}
    # soma だけ traub 型、dendrite はダミー型 traub_ (置換対象外)
    assert {n.name for n in ds.net.nodes if n.type.name == "traub"} == {"soma"}

    v = access.potential(
        unified_simulator(apply_surrogate(surrogate, ds.materialize())),
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
    assert replaceables(surrogate.meta, ds.net) == {"soma"}

    v = access.potential(
        unified_simulator(apply_surrogate(surrogate, ds.materialize())),
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
    source = surrogate.ansatz.train_source(surrogate.meta)
    gate = source.stacked_gate(surrogate.train_xr)
    ae_only = AEPreprocessor.fit(gate, surrogate.meta.n_components, {"epochs": 20})

    assert not np.allclose(
        surrogate.preprocessor.dec_params["W2"], ae_only.dec_params["W2"]
    )
    # 書き戻し後に再計算される派生物 (kernel の初期潜在・再構成指標) も joint 後の値
    assert len(surrogate.preprocessor.gate_inits) == surrogate.meta.n_components
    assert surrogate.preprocessor.reconstruction_mse != ae_only.reconstruction_mse


def test_ude_traub_transplants_across_heterogeneous_compartments() -> None:
    """UDE も hybrid と同じ kernel 骨格に載る (差分は潜在方程式の表現だけ) →
    Ca サブ系の physics 分離と traub19 全 comp への移植がそのまま成り立つ。"""
    surrogate = fit_surrogate("_test_traub_ude")
    assert surrogate.surr_comp_type.gate_names[-len(TRAUB_EXTRA_GATE_NAMES) :] == (
        TRAUB_EXTRA_GATE_NAMES
    )
    traub19 = SimSpec(target="traub19", current_type="train", dt=0.01)
    assert replaceables(surrogate.meta, traub19.net) == set(traub19.net.names)

    v = access.potential(
        unified_simulator(
            apply_surrogate(surrogate, surrogate.meta.dataset.materialize())
        ),
        _train_comp(surrogate),
    )
    assert np.isfinite(v).all()


def test_ude_rejects_non_learnable_preprocessor() -> None:
    """UDE は encoder/decoder を学習変数として更新する → 更新できない表現 (PCA) を
    渡されたら黙って前処理固定に退化せず、その場で落ちる。"""
    pca_based = fit_surrogate("_test_traub_hybrid")  # preprocessor_type=pca の preset
    with pytest.raises(ValueError, match="preprocessor_type=ae"):
        UDEAnsatz().fit(
            pca_based.meta, pca_based.train_xr, pca_based.preprocessor, {"epochs": 1}
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
    meta = surrogate.meta
    p = meta.train_comp.resolved_params
    assert isinstance(p, HHParams)

    # 潜在方程式 = 原 HH の dgate そのまま (latent==gate なので引数もそのまま渡せる)。
    def exact_dlatent(latent: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        v_rel = v - p.E_REST
        return jnp.stack(
            [dmdt(v_rel, latent[0]), dhdt(v_rel, latent[1]), dndt(v_rel, latent[2])]
        )

    surr_type = hybrid_surr_comp_type(
        meta,
        _IdentityPreprocessor(hh_inits(p)[1:]),  # 初期ゲート = 原 init のゲート部
        _NullClosure(),
        exact_dlatent,
    )
    replaced = dc_replace(
        meta.dataset.materialize(),
        net=replace_nodes(
            meta.dataset.net, surr_type, lambda n: n.type == meta.comp_type
        ),
    )

    comp = _train_comp(surrogate)
    assert np.array_equal(
        access.potential(unified_simulator(meta.dataset.materialize()), comp),
        access.potential(unified_simulator(replaced), comp),
    )


def test_hybrid_opcost_includes_decode() -> None:
    """hybrid の kernel は毎ステップ decode を呼ぶ → OpCost に計上されている。"""
    surrogate = fit_surrogate("_test_hh_hybrid")
    assert isinstance(surrogate.ansatz, HybridAnsatz)
    assert isinstance(surrogate.closure, SINDyBundle)  # opcost は表現固有
    decode_cost = surrogate.preprocessor.opcost()
    # PCA decode: gate ごとに latent 数の積 + 同数の加減 (3 latent x 3 gate)
    assert (decode_cost.mul, decode_cost.pm) == (9, 9)
    assert surrogate.surr_comp_type.opcost == (
        decode_cost
        + hybrid_physics(surrogate.meta).dv_cost
        + surrogate.closure.opcost()
    )
