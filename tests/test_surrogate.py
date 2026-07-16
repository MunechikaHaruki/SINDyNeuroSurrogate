"""サロゲート fit → 置換シミュ → 指標/描画の smoke (marimo/MLflow 非依存)。

Hydra プリセットを実設定源として読み、UI/実験ログを介さずドメイン層だけを通す。
"""

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from neurosurrogate.core import access
from neurosurrogate.metrics.eval import evaluate
from neurosurrogate.surrogate.ansatz import NeuroSurrogateBase

matplotlib.use("Agg")

CONF_DIR = Path(__file__).resolve().parents[1] / "scripts" / "conf"


def fit_surrogate(preset: str, n_components: int) -> NeuroSurrogateBase:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"surrogate={preset}",
                f"surrogate.init.n_components={n_components}",
            ],
        )
    c = OmegaConf.to_container(cfg.surrogate, resolve=True)
    assert isinstance(c, dict)
    surrogate = NeuroSurrogateBase.build(type=c["type"], init=c["init"])
    surrogate.fit(**c["fit"])
    return surrogate


@pytest.fixture(scope="module")
def sindy_surrogates() -> dict[int, NeuroSurrogateBase]:
    return {n: fit_surrogate("hh_sindy", n) for n in (1, 2, 3)}


@pytest.mark.parametrize("n_components", [1, 2, 3])
def test_sindy_replaced_sim_runs_at_any_latent_dim(sindy_surrogates, n_components):
    """列構造 [V, g1..gN, u] は latent 次元によらず置換シミュまで通る。"""
    surrogate = sindy_surrogates[n_components]
    assert surrogate.sindy_bundle.xi.shape[0] == n_components + 1  # V + latent
    assert len(surrogate.preprocessor_bundle.gate_inits) == n_components

    result = evaluate(surrogate, surrogate.meta.dataset)
    comp_id = surrogate.meta.train_comp_id
    v = access.potential(result.surr_ds, comp_id)
    assert v.shape == access.time(result.original_ds).shape
    assert np.isfinite(v[0])


def test_sindy_draws_all_figs(sindy_surrogates):
    result = evaluate(sindy_surrogates[2], sindy_surrogates[2].meta.dataset)
    from neurosurrogate.view.specs import draw_all

    assert [name for name, _ in draw_all(result, 0)] == ["diff", "simple", "attractor"]


def test_feature_exprs_align_with_xi_columns(sindy_surrogates):
    """feature 式列は xi の列と 1:1 (fit が pysindy 名との一致を検証済み)。"""
    bundle = sindy_surrogates[2].sindy_bundle
    assert len(bundle.feature_exprs) == bundle.xi.shape[1]


def test_duplicate_library_types_are_rejected(sindy_surrogates):
    """library type は互いに素 → 同 type 2 回で feature 式が重複しエラー。"""
    bundle = sindy_surrogates[2].sindy_bundle
    from neurosurrogate.surrogate.libraries.entry import FeatureLibrary

    library = FeatureLibrary.build(
        bundle.library_specs + bundle.library_specs, bundle.roles
    )
    with pytest.raises(ValueError, match="feature 重複"):
        library.bound_exprs(bundle.columns)


def test_equations_render_as_tex(sindy_surrogates):
    from neurosurrogate.view.model import equation_texs

    bundle = sindy_surrogates[2].sindy_bundle
    texs = equation_texs(bundle)
    assert len(texs) == len(bundle.targets)  # 1 target = 1 式
    assert all(t.startswith("$") and t.endswith("$") for t in texs)
    # 見出しは抜粋 → 先頭数項のみで残りは \cdots に畳む
    assert all(r"+ \cdots" in t for t in texs)
    # レート関数は未定義 Function → sympy が自動でギリシャ文字化。model は下付きへ
    # 回し、表示時に括弧へ整形 (mathtext が下付き内の空白を詰めるため)
    assert any(r"\alpha_{m(hh)}{\left(V \right)}" in t for t in texs)


def test_hybrid_opcost_includes_decode():
    """hybrid の kernel は毎ステップ decode を呼ぶ → OpCost に計上されている。"""
    surrogate = fit_surrogate("_hh_hybrid_pca_n3", 3)
    decode_cost = surrogate.preprocessor_bundle.opcost()
    # PCA decode: gate ごとに latent 数の積 + 同数の加減 (3 latent x 3 gate)
    assert (decode_cost.mul, decode_cost.pm) == (9, 9)
    assert surrogate.opcost == (
        decode_cost + surrogate._physics.dv_cost + surrogate.sindy_bundle.opcost()
    )
