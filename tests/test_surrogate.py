"""サロゲート fit → 置換シミュ → 指標/描画の smoke (marimo/MLflow 非依存)。

Hydra プリセットを実設定源として読み、UI/実験ログを介さずドメイン層だけを通す。
設定は `conf/surrogate/_test_*.yaml` (素体から library_specs を継承し、学習構造と
短縮電流だけ固定したテスト専用プリセット) に置き、テスト側は override しない。
"""

import json
from dataclasses import replace as dc_replace
from functools import cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from matplotlib.figure import Figure
from omegaconf import OmegaConf

from neurosurrogate.core import access
from neurosurrogate.core.network import DatasetConfig
from neurosurrogate.core.opcost import OpCost
from neurosurrogate.core.simulator import unified_simulator
from neurosurrogate.eval.eval import (
    EvalGrid,
    EvalPoint,
    evaluate,
    preprocessed_latent,
)
from neurosurrogate.eval.spec import EvalSpec, SweepAxis, parse_evals
from neurosurrogate.eval.store import artifacts, load_all, save
from neurosurrogate.metrics.engine import collect, new_figure
from neurosurrogate.metrics.figs.cell import cell_figs, panels_simple
from neurosurrogate.metrics.figs.grid import compare_grid_fig, trace_grid_fig
from neurosurrogate.metrics.figs.model import equation_texs, preprocessor_figs
from neurosurrogate.metrics.figs.train import train_figs
from neurosurrogate.metrics.report import (
    CompareSpec,
    DrawSpec,
    ReportSpec,
    ResultSpec,
    eval_report,
)
from neurosurrogate.metrics.wave import METRIC_KEYS, DynamicMetrics, extract_metric
from neurosurrogate.neurons.compartments.hh import HHParams, dhdt, dmdt, dndt, hh_inits
from neurosurrogate.neurons.compartments.traub import (
    TRAUB_EXTRA_GATE_NAMES,
    TRAUB_SR_EXTRA_GATE_NAMES,
)
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
from neurosurrogate.surrogate.preprocessor.base import Preprocessor
from neurosurrogate.surrogate.preprocessor.impl.autoencoder import AEPreprocessor
from neurosurrogate.surrogate.preprocessor.impl.pca import PCAPreprocessor
from neurosurrogate.surrogate.replace import (
    apply_surrogate,
    replace_nodes,
    replaceables,
)

CONF_DIR = Path(__file__).resolve().parents[1] / "scripts" / "conf"
LATENT_DIMS = [1, 3]  # 単一 latent と複数 latent = 列構造 [V, g1..gN, u] の両端


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


def _spec_of(bundle: SurrogateBundle) -> EvalSpec:
    """学習データと同じ入力の評価仕様 (掃引軸なし = 点 1 つ)。"""
    ds = bundle.meta.dataset
    return EvalSpec(
        target=ds.model_name,
        current_type=ds.current_type,
        dt=ds.dt,
        current_params=ds.current_params,
    )


@pytest.fixture(scope="module")
def sindy_grid(sindy: SurrogateBundle) -> EvalGrid:
    """単発 = 点 1 つ・run 1 本の退化グリッド (掃引と同じ型・同じ経路)。"""
    return evaluate({"r0": sindy}, _spec_of(sindy))


@pytest.fixture(scope="module")
def sindy_closure(sindy: SurrogateBundle) -> SINDyBundle:
    """ξ / feature 式は SINDy 固有 (bundle.closure は表現非依存の Closure 型)。"""
    assert isinstance(sindy.closure, SINDyBundle)
    return sindy.closure


@pytest.mark.parametrize("n_components", LATENT_DIMS)
def test_sindy_replaced_sim_runs_at_any_latent_dim(n_components: int) -> None:
    """列構造 [V, g1..gN, u] は latent 次元によらず置換シミュまで通る。"""
    surrogate = fit_surrogate("_test_hh_sindy", n_components)
    assert isinstance(surrogate.closure, SINDyBundle)
    assert surrogate.closure.xi.shape[0] == n_components + 1  # V + latent
    assert len(surrogate.preprocessor.gate_inits) == n_components

    point = evaluate({"r0": surrogate}, _spec_of(surrogate)).points[0]
    v = access.potential(point.surrogates["r0"], _train_comp(surrogate))
    assert v.shape == access.time(point.original).shape
    assert np.isfinite(v[0])


def test_sweep_metric_choices_are_all_extractable(sindy_grid: EvalGrid) -> None:
    """UI が出す掃引 metric 選択肢は全て取り出せる = 選んだのに生成されないキーで
    黙って nan の図が出ることが無い (未知キーは extract_metric が KeyError)。"""
    point = sindy_grid.points[0]
    dm = DynamicMetrics(point.original, point.surrogates["r0"], 0, sindy_grid.spec.dt)
    assert all(extract_metric(dm, key)[1] is not None for key in METRIC_KEYS)
    with pytest.raises(KeyError):
        extract_metric(dm, "latency_error")


def test_sindy_draws_all_figs(sindy_grid: EvalGrid, sindy: SurrogateBundle) -> None:
    """1 セル (点 × run) の詳細図。潜在射影は callable で遅延評価される。"""
    point = sindy_grid.points[0]
    figs = cell_figs(
        point.original,
        point.surrogates["r0"],
        0,
        lambda: preprocessed_latent(sindy, sindy_grid.spec.net, point.original, 0),
    )
    assert [name for name, _ in figs] == ["diff", "simple", "attractor"]


def test_eval_and_draw_json_are_self_consistent() -> None:
    """marimo/CLI の既定設定が自己整合: `eval.json` の全 entry の電流が掃引点まで
    含めて構築でき、`draw.json` の `results`/`compare` が参照する label は
    `eval.json` の label に実在する (2 ファイルに分けたことで生まれうる typo/ズレを
    テストで担保する)。単発 entry も「点 1 つ」として同じ経路を通る。"""
    conf_dir = Path(__file__).parents[1] / "scripts/conf"
    evals = parse_evals(json.loads((conf_dir / "eval.json").read_text()))
    for spec in evals.values():
        for point in spec.points:
            assert len(spec.dataset_at(point).build_current()) > 0
    assert [len(s.points) for s in evals.values()][0] == 1  # 掃引軸なし = 点 1 つ

    report = ReportSpec.from_dict(json.loads((conf_dir / "draw.json").read_text()))
    assert {r.label for r in report.results} <= set(evals)
    for comparison in report.compares.values():
        assert set(comparison.evals) <= set(evals)


def test_compare_grid_rows_are_current_then_one_per_eval(
    sindy: SurrogateBundle,
) -> None:
    """compare 図の行 = [I_ext] + [評価ごとの V]、列 = 点。点数が揃わない結果を
    混ぜると列の意味が行ごとにずれる → raise。"""
    spec = EvalSpec(
        target="hh",
        current_type="lin&steady",
        dt=0.05,
        current_params={"duration": 30.0, "silence_duration": 0.0},
        sweep=SweepAxis(param="value", start=5.0, stop=10.0, steps=2),
    )
    grid = evaluate({"r0": sindy}, spec)
    fig = compare_grid_fig({"a": grid, "b": grid}, "soma")
    assert len(fig.axes) == 3 * 2  # (I_ext + a + b) 行 × 2 点
    assert [ax.get_ylabel() for ax in fig.axes[::2]] == ["I_ext", "a", "b"]

    # 同じ格子骨格を run 軸で開くと行 = [I_ext] + [run] (行の組み方だけが違う)。
    run_fig = trace_grid_fig(grid, "soma")
    assert [ax.get_ylabel() for ax in run_fig.axes[::2]] == ["I_ext", "r0"]

    axis = SweepAxis(param="value", start=5.0, stop=10.0, steps=3)
    short = evaluate({"r0": sindy}, dc_replace(spec, sweep=axis))
    with pytest.raises(ValueError, match="点数"):
        compare_grid_fig({"a": grid, "b": short}, "soma")


def test_result_artifacts_round_trip_without_resimulating(
    sindy_grid: EvalGrid, tmp_path: Path
) -> None:
    """結果 artifact = **1 surrogate run × 1 spec**。保存 → 読込で再シミュ無しに
    同じ波形が戻り、run 軸ごとに分かれて保存され読込で束ね直る。artifact に
    surrogate は焼き込まず出所 run_id だけを持つ。"""
    root = tmp_path / "artifacts"
    root.mkdir()
    n = len(save(sindy_grid, "hh_dc", root, {"r0": "RID"}, "PARENT"))
    assert n == 1  # run 軸 1 本

    # 同じ label で入力仕様だけ変えて回し直した系列 (束ねたら点の意味がずれる)
    point = sindy_grid.points[0]
    other = EvalGrid(
        spec=dc_replace(sindy_grid.spec, dt=sindy_grid.spec.dt * 2),
        points=[EvalPoint(None, point.original, {"r1": point.surrogates["r0"]})],
    )
    save(other, "hh_dc", root, {"r1": "RID"}, "PARENT")

    arts = artifacts(root)
    assert {a.meta.run_id for a in arts} == {"RID"}  # surrogate でなく run_id を持つ

    # label でなく (label, 入力仕様) で束ね、label 衝突は新しい系列が勝つ
    loaded = load_all(arts)
    assert list(loaded) == ["hh_dc"]
    assert loaded["hh_dc"].run_labels == ["r1"]
    assert loaded["hh_dc"].spec.dt == other.spec.dt
    # 入力仕様から dataset を復元でき、波形は float32 で往復する
    assert loaded["hh_dc"].spec.dataset_at(None).model_name == sindy_grid.spec.target
    np.testing.assert_allclose(
        access.potential(loaded["hh_dc"].points[0].surrogates["r1"], 0),
        access.potential(point.surrogates["r0"], 0),
        rtol=1e-5,
    )


def test_report_draws_the_results_at_hand_not_the_declaration(
    sindy_grid: EvalGrid, sindy: SurrogateBundle
) -> None:
    """描画は**手元の結果だけ**を見る (計算入力の設定と突き合わせない): 設定ファイル
    に宣言の無い label — 別セッションで回して artifact から読んだ結果 — もそのまま
    図になり、逆に参照先が手元に無い compare は error 図でなく**黙って落ちる**
    (宣言とのズレは呼び出し側の関心) = 計算と描画が切れている。"""
    report = ReportSpec(
        results=(ResultSpec(label="読んだ系列", draw=DrawSpec(eval_comp="soma")),)
    )
    entries = eval_report({"読んだ系列": sindy_grid}, {"r0": sindy}, report)
    assert any(e.name.startswith("読んだ系列/") for e in entries)

    dangling = CompareSpec(name="c", evals=["未実行"], eval_comp="soma")
    report_with_compare = ReportSpec(compares={"c": dangling})
    assert eval_report({}, {"r0": sindy}, report_with_compare) == []


def test_report_spec_results_are_per_label_with_no_default_fallback() -> None:
    """`draw.json` の `results[]` は label ごとに完結する宣言 (既定値からの override
    ではない): 指定したキーだけ効き、欠落キーは `DrawSpec` の型既定値。宣言に無い
    label は `DrawSpec()` そのもの (グローバル既定を持たない)。"""
    report = ReportSpec.from_dict(
        {"results": [{"eval": "traub19_dendstim", "eval_comp": "c09"}]}
    )
    assert report.draw_for("traub19_dendstim") == DrawSpec(eval_comp="c09")
    assert report.draw_for("宣言に無い label") == DrawSpec()


def test_draw_settings_are_typed_and_failed_figs_fold_into_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """表示設定は widget/保存 dict を読む型 1 つが源 (欠落キーは既定値)。描画 job の
    失敗は列を保ったまま error 図へ畳む = 1 図の失敗で他の図まで落とさない。"""
    assert DrawSpec.from_dict({}).metric_ylim() is None  # 既定は y auto
    drawn = DrawSpec.from_dict(
        {"eval_comp": "soma", "metric_yauto": False, "metric_ymax": 40}
    )
    assert (drawn.eval_comp, drawn.metric_ylim()) == ("soma", (0.0, 40.0))

    def boom() -> Figure:
        raise KeyError("missing var")

    assert [name for name, _ in collect({"ok": new_figure, "ng": boom})] == ["ok", "ng"]
    assert "ng" in capsys.readouterr().err  # 失敗は握り潰さず stderr へも出す


def test_view_comps_limit_drawn_traces(
    sindy_grid: EvalGrid, sindy: SurrogateBundle
) -> None:
    """表示 comp 制限 (UI の view_comps) が全 comp を並べる図に効く: 対象外だけを
    指定するとパネル/trace が消え、学習 comp を指定した学習データ図は描ける。"""
    ds = sindy_grid.points[0].original
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


def test_equations_render_as_tex(sindy_closure: SINDyBundle) -> None:
    texs = equation_texs(sindy_closure)
    assert len(texs) == len(sindy_closure.targets)  # 1 target = 1 式
    assert all(t.startswith("$") and t.endswith("$") for t in texs)
    # 見出しは抜粋 → 先頭数項のみで残りは \cdots に畳む
    assert all(r"+ \cdots" in t for t in texs)
    # レート関数は未定義 Function → sympy が自動でギリシャ文字化。model は下付きへ
    # 回し、表示時に括弧へ整形 (mathtext が下付き内の空白を詰めるため)
    assert any(r"\alpha_{m(hh)}{\left(V \right)}" in t for t in texs)


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

    traub19 = DatasetConfig.build_dataset(
        dt=0.01, model_name="traub19", current_type="train", current_params={}
    )
    # phi_area/g_Ca が異なる 19 comp すべてが置換対象。pre-B は soma のみ一致で
    # ValueError だった (Ca params が latent に焼込まれ params 一致必須だったため)。
    assert replaceables(surrogate.meta, traub19) == set(traub19.net.names)

    # 置換シミュ (XI/Q を各ノード params で physics 積分) が有限に走る。
    v = access.potential(
        unified_simulator(apply_surrogate(surrogate, surrogate.meta.dataset)),
        _train_comp(surrogate),
    )
    assert np.isfinite(v).all()


def test_traub19_soma_model_replaces_only_soma() -> None:
    """適用先モデル traub19_soma は soma だけ traub 型に残し dendrite をダミー型に
    する → comp_type=traub の学習を **preset 変更なし** で soma 1 ノードだけへ適用
    できる (置換範囲を絞る新軸を meta へ足さず、適用先モデル側で絞る)。dendrite 18 個
    は置換対象外のまま残り、置換シミュが有限に走る。"""
    surrogate = fit_surrogate("_test_traub_hybrid")  # comp_type=traub, 単体 traub 教師
    ds = DatasetConfig.build_dataset(
        dt=0.01,
        model_name="traub19_soma",
        current_type="train",
        current_params={"duration": 180},  # smoke: 配線確認のみ (本番は長時間)
    )
    assert replaceables(surrogate.meta, ds) == {"soma"}
    # soma だけ traub 型、dendrite はダミー型 traub_ (置換対象外)
    assert {n.name for n in ds.net.nodes if n.type.name == "traub"} == {"soma"}

    v = access.potential(
        unified_simulator(apply_surrogate(surrogate, ds)),
        ds.net.name_to_idx("soma"),
    )
    assert np.isfinite(v).all()


def test_traub19_soma_dendstim_injects_into_dendrite() -> None:
    """dend 刺激版も soma だけ置換対象 (traub 型 = soma のみ) だが、電流注入先は
    dendrite。刺激点が soma でないこと + 置換シミュが有限に走ることを確認。"""
    surrogate = fit_surrogate("_test_traub_hybrid")
    ds = DatasetConfig.build_dataset(
        dt=0.01,
        model_name="traub19_soma_dendstim",
        current_type="train",
        current_params={"duration": 180},
    )
    assert ds.net.stim != "soma"  # 注入先は dendrite
    assert replaceables(surrogate.meta, ds) == {"soma"}

    v = access.potential(
        unified_simulator(apply_surrogate(surrogate, ds)),
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
    traub19 = DatasetConfig.build_dataset(
        dt=0.01, model_name="traub19", current_type="train", current_params={}
    )
    assert replaceables(surrogate.meta, traub19) == set(traub19.net.names)

    v = access.potential(
        unified_simulator(apply_surrogate(surrogate, surrogate.meta.dataset)),
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
        meta.dataset,
        net=replace_nodes(
            meta.dataset.net, surr_type, lambda n: n.type == meta.comp_type
        ),
    )

    comp = _train_comp(surrogate)
    assert np.array_equal(
        access.potential(unified_simulator(meta.dataset), comp),
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
