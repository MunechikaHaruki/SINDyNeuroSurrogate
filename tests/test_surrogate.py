"""サロゲート fit → 置換シミュ → 指標/描画の smoke (marimo/MLflow 非依存)。

Hydra プリセットを実設定源として読み、UI/実験ログを介さずドメイン層だけを通す。
設定は `conf/surrogate/_test_*.yaml` (素体から library_specs を継承し、学習構造と
短縮電流だけ固定したテスト専用プリセット) に置き、テスト側は override しない。
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from neurosurrogate.core import access
from neurosurrogate.core.network import DatasetConfig
from neurosurrogate.core.simulator import unified_simulator
from neurosurrogate.metrics.eval import EvalResult, evaluate
from neurosurrogate.surrogate.ansatz.hybrid import HybridAnsatz
from neurosurrogate.surrogate.bundle import SurrogateBundle
from neurosurrogate.surrogate.closure.sindy import SINDyBundle
from neurosurrogate.surrogate.closure.sindy.entry import FeatureLibrary
from neurosurrogate.surrogate.replace import apply_surrogate, replaceables
from neurosurrogate.view.model import equation_texs
from neurosurrogate.view.specs import draw_all, spec_simple
from neurosurrogate.view.train import train_figs

CONF_DIR = Path(__file__).resolve().parents[1] / "scripts" / "conf"
LATENT_DIMS = [1, 3]  # 単一 latent と複数 latent = 列構造 [V, g1..gN, u] の両端


def fit_surrogate(preset: str, n_components: int | None = None) -> SurrogateBundle:
    """テスト専用プリセットを fit。テストの唯一の surrogate 生成口。
    n_components だけは preset 既定を上書きできる (列構造を振るテストのため)。"""
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


@pytest.fixture(scope="module")
def sindy_eval(sindy: SurrogateBundle) -> EvalResult:
    return evaluate(sindy, sindy.meta.dataset)


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

    result = evaluate(surrogate, surrogate.meta.dataset)
    v = access.potential(result.surr_ds, _train_comp(surrogate))
    assert v.shape == access.time(result.original_ds).shape
    assert np.isfinite(v[0])


def test_sindy_draws_all_figs(sindy_eval: EvalResult) -> None:
    assert [name for name, _ in draw_all(sindy_eval, 0)] == [
        "diff",
        "simple",
        "attractor",
    ]


def test_view_comps_limit_drawn_traces(
    sindy_eval: EvalResult, sindy: SurrogateBundle
) -> None:
    """表示 comp 制限 (UI の view_comps) が全 comp を並べる図に効く: 対象外だけを
    指定するとパネル/trace が消え、学習 comp を指定した学習データ図は描ける。"""
    ds = sindy_eval.original_ds
    assert len(spec_simple(ds, comps=[])) < len(spec_simple(ds))
    assert [name for name, _ in train_figs(sindy, comps=[_train_comp(sindy)])] == [
        name for name, _ in train_figs(sindy)
    ]


def test_train_figs_render_from_reloaded_surrogate(
    sindy: SurrogateBundle, tmp_path: Path
) -> None:
    """学習データ図は save/load を跨いで描ける: 軌道は保存されず meta +
    ansatz.train_source から再生成される (marimo が run ロード毎に描く経路)。"""
    sindy.save(tmp_path)
    # meta は JSON で別ファイル → 一覧側は pickle を開かずに同定情報を読める
    assert SurrogateBundle.load_meta(tmp_path) == sindy.meta
    reloaded = SurrogateBundle.load(tmp_path)
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
    # Ca サブ系 (XI/Q) は学習に含めない
    assert source.n_gate == len(surrogate.meta.comp_type.gate_names) - 2
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


def test_hybrid_traub_transplants_across_heterogeneous_compartments() -> None:
    """hybrid traub は Ca サブ系 (XI/Q) を physics へ分離し純電位依存ゲートのみ学習 →
    Ca params (phi_area/g_Ca) がノード毎に違う traub19 全 comp を 1 サロゲートで置換
    できる (compatible=True)。学習は 8 電位依存ゲートのみ、surr state は latent+[XI,Q]。
    新 kernel の XI/Q 積分が置換シミュまで有限に走ることも確認する。
    preprocessor は AE の乱数初期化で fit 品質 (=有限性) がブレる → 決定的な pca に固定
    (主眼は XI/Q physics 積分経路であり AE 再構成品質ではない)。"""
    surrogate = fit_surrogate("_test_traub_hybrid")
    assert surrogate.surr_comp_type.gate_names[-2:] == ["XI", "Q"]

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


def test_hybrid_opcost_includes_decode() -> None:
    """hybrid の kernel は毎ステップ decode を呼ぶ → OpCost に計上されている。"""
    surrogate = fit_surrogate("_test_hh_hybrid")
    ansatz = surrogate.ansatz
    assert isinstance(ansatz, HybridAnsatz)  # _physics は hybrid 固有
    assert isinstance(surrogate.closure, SINDyBundle)  # opcost は表現固有
    decode_cost = surrogate.preprocessor.opcost()
    # PCA decode: gate ごとに latent 数の積 + 同数の加減 (3 latent x 3 gate)
    assert (decode_cost.mul, decode_cost.pm) == (9, 9)
    assert surrogate.opcost == (
        decode_cost
        + ansatz._physics(surrogate.meta).dv_cost
        + surrogate.closure.opcost()
    )
