"""サロゲート fit → 置換シミュ → 指標/描画の smoke (marimo/MLflow 非依存)。

Hydra プリセットを実設定源として読み、UI/実験ログを介さずドメイン層だけを通す。
学習電流は smoke 用に短縮 (`TRAIN_DURATION`) — 波形パラメータは本番と同一。
"""

from pathlib import Path

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
from neurosurrogate.view.specs import draw_all

CONF_DIR = Path(__file__).resolve().parents[1] / "scripts" / "conf"
TRAIN_DURATION = 180  # [ms] 本番は 9000。smoke は shape/有限性のみ見るので短縮
LATENT_DIMS = [1, 3]  # 単一 latent と複数 latent = 列構造 [V, g1..gN, u] の両端
REPRESENTATIVE_DIM = 3  # 式構造/描画テストの代表。latent 複数の方が構造が厳しい
# preset default の ansatz/preprocessor は sweep 軸で振れる → 明示 override する
SINDY = ["surrogate.meta.surrogate_type=sindy"]
HYBRID_PCA = [
    "surrogate.meta.surrogate_type=hybrid",
    "surrogate.meta.preprocessor_type=pca",
]


def fit_surrogate(
    preset: str, n_components: int, extra: list[str] | None = None
) -> SurrogateBundle:
    """Hydra プリセットを短縮電流で fit。テストの唯一の surrogate 生成口。"""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"surrogate={preset}",
                f"surrogate.meta.n_components={n_components}",
                f"+surrogate.meta.datasets.current_params.duration={TRAIN_DURATION}",
                *(extra or []),
            ],
        )
    c = OmegaConf.to_container(cfg.surrogate, resolve=True)
    assert isinstance(c, dict)
    return SurrogateBundle.setup(c)


@pytest.fixture(scope="module")
def sindy() -> SurrogateBundle:
    """代表 sindy surrogate。latent 次元に依らない性質のテストが共有する。"""
    return fit_surrogate("hh", REPRESENTATIVE_DIM, extra=SINDY)


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
    surrogate = fit_surrogate("hh", n_components, extra=SINDY)
    assert isinstance(surrogate.closure, SINDyBundle)
    assert surrogate.closure.xi.shape[0] == n_components + 1  # V + latent
    assert len(surrogate.preprocessor.gate_inits) == n_components

    result = evaluate(surrogate, surrogate.meta.dataset)
    v = access.potential(result.surr_ds, surrogate.meta.train_comp_id)
    assert v.shape == access.time(result.original_ds).shape
    assert np.isfinite(v[0])


def test_sindy_draws_all_figs(sindy_eval: EvalResult) -> None:
    assert [name for name, _ in draw_all(sindy_eval, 0)] == [
        "diff",
        "simple",
        "attractor",
    ]


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
    surrogate = fit_surrogate(
        "traub",
        5,
        extra=HYBRID_PCA,
    )
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
        surrogate.meta.train_comp_id,
    )
    assert np.isfinite(v).all()


def test_hybrid_opcost_includes_decode() -> None:
    """hybrid の kernel は毎ステップ decode を呼ぶ → OpCost に計上されている。"""
    surrogate = fit_surrogate(
        "hh",
        3,
        extra=HYBRID_PCA,
    )
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
