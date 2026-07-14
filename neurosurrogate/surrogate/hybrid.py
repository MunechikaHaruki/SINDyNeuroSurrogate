import jax.numpy as jnp

from ..compartments.hh import HH_DV_COST, HHParams
from ..core import access
from ..core.network import Compartment, CompartmentType
from ..core.opcost import OpCost
from . import get_gate_numpy
from .base import NeuroSurrogateBase
from .bundle import PreprocessorBundle, SINDyBundle
from .replace import resolved_params


class HybridSINDyNeuroSurrogate(NeuroSurrogateBase):
    """Hybrid: 物理HH回路方程式 (HHParams 陽) + 学習ゲート dynamics。
      dV/dt   = HH 物理式 (E_LEAK, G_*, E_*, C, gates)
      gates   = state @ pca_components + pca_mean    (線形 decoder)
      d(latent)/dt = f(V, latent) via SINDy
    SINDy 入力順: (latent_1, ..., V) → library_specs は index 0=latent, 末尾=V。
    """

    SURROGATE_TYPE = "hybrid"

    def fit(self, preprocessor, optimizer, library_specs: list[dict]) -> None:
        gate_data = get_gate_numpy(self._train_xr, self._meta.train_comp_id)
        preprocessor_bundle = PreprocessorBundle.from_spec(preprocessor, gate_data)
        latent = preprocessor_bundle.preprocessor.transform(gate_data)
        v = access.potential(self._train_xr, self._meta.train_comp_id)
        latent_names = [f"latent{i + 1}" for i in range(latent.shape[1])]
        self._set_bundles(
            sindy_bundle=SINDyBundle.from_sindy(
                library_specs=library_specs,
                optimizer_spec=optimizer,
                x=latent,
                u=v,
                t=access.time(self._train_xr),
                target_names=latent_names,
                input_names=["V"],
            ),
            preprocessor_bundle=preprocessor_bundle,
        )

    def params_compatible(self, comp: Compartment) -> bool:
        # 物理 dV/dt (hybrid_kernel) はノード params (G_*, E_*, C 等) を陽に読む
        # → 導電率・反転電位は自由。一方 gate dynamics は SINDy が絶対 V で fit し
        # 学習ノードの E_REST を暗黙に焼込む (rate は v_rel=v-E_REST 依存) →
        # E_REST のみ一致必須。
        return bool(
            resolved_params(self.meta.train_comp).E_REST  # type: ignore[union-attr]
            == resolved_params(comp).E_REST  # type: ignore[union-attr]
        )

    @property
    def surr_comp_type(self) -> CompartmentType:
        xi = jnp.asarray(self.sindy_bundle.xi)
        bundle = self.preprocessor_bundle.bundle
        assert bundle is not None
        decode = bundle.decode
        compute_theta = self.sindy_bundle.compute_theta()
        n_latent = len(self.preprocessor_bundle.gate_inits)

        def hybrid_kernel(p: HHParams, u_t, v, state):
            gates = decode(state)
            m, h, n = gates[0], gates[1], gates[2]
            i_na = p.G_NA * m**3 * h * (v - p.E_NA)
            i_k = p.G_K * n**4 * (v - p.E_K)
            i_leak = p.G_LEAK * (v - p.E_LEAK)
            dv = (-i_leak - i_na - i_k + u_t) / p.C
            theta_inputs = [state[i] for i in range(n_latent)] + [v]
            theta = compute_theta(*theta_inputs)
            dlatent = xi @ theta
            return dv, dlatent

        return CompartmentType(
            name="hybrid_surr",
            kernel=hybrid_kernel,
            param_cls=HHParams,
            gate_names=[f"latent{i + 1}" for i in range(n_latent)],
            default_gate_inits=self.preprocessor_bundle.gate_inits,
            v_init=-65,
            opcost=None,
        )

    @property
    def opcost(self) -> OpCost:
        return self.sindy_bundle.opcost() + HH_DV_COST
