from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import sympy as sp

from ...compartments.hh import HH_DV_COST, HHParams, hh_dv
from ...compartments.traub import TRAUB_DV_COST, TRAUB_V_INIT, TraubParams, traub_dv
from ...core import access
from ...core.network import Compartment, CompartmentType
from ...core.opcost import OpCost
from ..bundle import PreprocessorBundle, SINDyBundle
from ..replace import resolved_params
from .base import NeuroSurrogateBase
from .roles import Roles


@dataclass(frozen=True)
class HybridPhysics:
    """学習型ごとの物理 dV/dt 差分を集約 (hybrid で dispatch する単位)。

    dv          : (params, u_t, v, gates) -> dv。gates は decode 済 latent。
    dv_cost     : dv の演算コスト。
    v_init      : 置換後 surrogate の初期電位。
    compatible  : (train_params, node_params) -> 置換両立か。
                  hybrid の dV/dt は params を陽に読むが、gate/latent dynamics は
                  SINDy が学習ノードの物理を暗黙に焼込む → 焼込む params のみ一致必須。
    """

    param_cls: type
    dv: Callable
    dv_cost: OpCost
    v_init: float
    compatible: Callable[[tuple, tuple], bool]


HYBRID_PHYSICS: dict[str, HybridPhysics] = {
    # HH: gate rate は v_rel=v-E_REST 依存 → E_REST のみ一致必須。G_*/E_*/C は
    # dv が陽に読むため自由。
    "hh": HybridPhysics(
        param_cls=HHParams,
        dv=hh_dv,
        dv_cost=HH_DV_COST,
        v_init=-65.0,
        compatible=lambda a, b: bool(a.E_REST == b.E_REST),  # type: ignore[attr-defined]
    ),
    # Traub: gate rate は定数 TRAUB_V_LEAK 基準で params 非依存だが、XI(Ca) dynamics
    # が i_ca(g_Ca,V_Ca)・Beta・phi_area を latent へ焼込む
    # → 保守的に全 params 一致必須。
    "traub": HybridPhysics(
        param_cls=TraubParams,
        dv=traub_dv,
        dv_cost=TRAUB_DV_COST,
        v_init=TRAUB_V_INIT,
        compatible=lambda a, b: a == b,
    ),
}


class HybridSINDyNeuroSurrogate(NeuroSurrogateBase):
    """Hybrid: 物理回路方程式 (params 陽) + 学習ゲート dynamics。
      dV/dt   = 学習型の物理式 (HYBRID_PHYSICS[train_type].dv)
      gates   = decode(latent)                        (線形/AE decoder)
      d(latent)/dt = f(V, latent) via SINDy
    SINDy 入力順: (g1, ..., V) → library_specs は index 0=latent, 末尾=V。
    HH/Traub 両対応 (gate 数・param_cls・dv は学習型から解決)。
    """

    SURROGATE_TYPE = "hybrid"

    @property
    def _physics(self) -> HybridPhysics:
        return HYBRID_PHYSICS[self.meta.train_comp_type.name]

    def fit(self, preprocessor, optimizer, library_specs: list[dict]) -> None:
        gate_data = access.gate_matrix(self._train_xr, self._meta.train_comp_id)
        preprocessor_bundle = PreprocessorBundle.from_spec(
            {**preprocessor, "n_components": self._n_components}, gate_data
        )
        latent = preprocessor_bundle.preprocessor.transform(gate_data)
        self._set_bundles(
            sindy_bundle=SINDyBundle.from_sindy(
                library_specs=library_specs,
                optimizer_spec=optimizer,
                x=latent,
                u=access.potential(self._train_xr, self._meta.train_comp_id),
                t=access.time(self._train_xr),
                targets=[sp.Symbol(v) for v in access.latent_vars(self._n_components)],
                inputs=[sp.Symbol(access.POTENTIAL_VAR)],
                # 列構造: [g1..gN, V]。gate 群が先頭、末尾に V。u は入力に無し。
                roles=Roles(V=self._n_components, g=list(range(self._n_components))),
            ),
            preprocessor_bundle=preprocessor_bundle,
        )

    def params_compatible(self, comp: Compartment) -> bool:
        return self._physics.compatible(
            resolved_params(self.meta.train_comp),  # type: ignore[arg-type]
            resolved_params(comp),  # type: ignore[arg-type]
        )

    @property
    def surr_comp_type(self) -> CompartmentType:
        phys = self._physics
        xi = jnp.asarray(self.sindy_bundle.xi)
        bundle = self.preprocessor_bundle.bundle
        assert bundle is not None
        decode = bundle.decode
        compute_theta = self.sindy_bundle.compute_theta()
        n_latent = len(self.preprocessor_bundle.gate_inits)

        def hybrid_kernel(p, u_t, v, state):
            gates = decode(state)
            dv = phys.dv(p, u_t, v, gates)
            theta_inputs = [state[i] for i in range(n_latent)] + [v]
            theta = compute_theta(*theta_inputs)
            dlatent = xi @ theta
            return dv, dlatent

        return CompartmentType(
            name="hybrid_surr",
            kernel=hybrid_kernel,
            param_cls=phys.param_cls,
            gate_names=access.latent_vars(n_latent),
            default_gate_inits=self.preprocessor_bundle.gate_inits,
            v_init=phys.v_init,
            opcost=None,
        )

    @property
    def opcost(self) -> OpCost:
        # kernel の 1 ステップ = decode(latent→gate) + 物理 dV/dt + SINDy 潜在方程式。
        return (
            self.preprocessor_bundle.opcost()
            + self._physics.dv_cost
            + self.sindy_bundle.opcost()
        )
