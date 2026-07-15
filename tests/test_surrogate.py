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
    """列構造 [V, latent1..N, u] は latent 次元によらず置換シミュまで通る。"""
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


def test_hybrid_opcost_includes_decode():
    """hybrid の kernel は毎ステップ decode を呼ぶ → OpCost に計上されている。"""
    surrogate = fit_surrogate("_hh_hybrid_pca_n3", 3)
    decode_cost = surrogate.preprocessor_bundle.opcost()
    # PCA decode: gate ごとに latent 数の積 + 同数の加減 (3 latent x 3 gate)
    assert (decode_cost.mul, decode_cost.pm) == (9, 9)
    assert surrogate.opcost == (
        decode_cost + surrogate._physics.dv_cost + surrogate.sindy_bundle.opcost()
    )
