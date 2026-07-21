from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import sympy as sp

from ...compartments.hh import HH_DV_COST, HHParams, hh_dv
from ...compartments.traub import (
    TRAUB_CA_COST,
    TRAUB_DV_COST,
    TRAUB_EXTRA_GATE_NAMES,
    TRAUB_LEARNED_GATE_NAMES,
    TraubParams,
    traub_calcium_step,
    traub_dv,
    traub_extra_inits,
)
from ...core import access
from ...core.network import Compartment, CompartmentType
from ...core.opcost import OpCost
from ..bundle import SINDyBundle
from .base import NeuroSurrogateBase
from .roles import Roles


@dataclass(frozen=True)
class ExtraPhysics:
    """学習 latent から外し physics で解く追加状態 (Traub の Ca サブ系 XI/Q)。

    これらの dynamics は params を陽に読む (置換先ノード自身の p で解く) → 学習には
    含めない。含めれば params が latent へ焼込まれ compatible が params 一致を要求
    してしまう。physics 化することで学習は純 params-free ゲートのみになる。

    names : surr state で latent の後へ付く状態名 (gate_names 追加分・init 順)。
    step  : (p, v, gates_learned, extra_state) -> (dv 用の全ゲート, d(extra))。
    inits : (p) -> extra 各状態の初期値 (params のみ依存 → load 後も再現可)。
    cost  : extra step の演算コスト。
    """

    names: list[str]
    step: Callable
    inits: Callable[[Any], list[float]]
    cost: OpCost


@dataclass(frozen=True)
class HybridPhysics:
    """学習型ごとの物理 dV/dt 差分を集約 (hybrid で dispatch する単位)。

    dv           : (params, u_t, v, gates) -> dv。gates は dv 用の全ゲート。
    dv_cost      : dv の演算コスト。
    v_init       : (params) -> 置換後 surrogate の初期電位 (ノード自身の静止電位)。
    n_learned    : 学習 latent が圧縮する (先頭) ゲート数。
    extra        : physics で解く追加状態 (無ければ None → 学習ゲート=全ゲート)。
    compatible   : (train_params, node_params) -> 置換両立か。物理 dv も extra も
                   params を陽に読むため、latent へ焼込まれる params のみ一致必須。
    """

    param_cls: type
    dv: Callable
    dv_cost: OpCost
    v_init: Callable[[Any], float]
    n_learned: int
    extra: ExtraPhysics | None
    compatible: Callable[[tuple, tuple], bool]


HYBRID_PHYSICS: dict[str, HybridPhysics] = {
    # HH: 3 ゲート全てが純電位依存 (Ca 無し) → extra 無し。gate rate は v_rel=v-E_REST
    # 依存 → E_REST のみ一致必須。G_*/E_*/C は dv が陽に読むため自由。
    "hh": HybridPhysics(
        param_cls=HHParams,
        dv=hh_dv,
        dv_cost=HH_DV_COST,
        v_init=lambda p: p.E_REST,
        n_learned=3,
        extra=None,
        compatible=lambda a, b: bool(a.E_REST == b.E_REST),  # type: ignore[attr-defined]
    ),
    # Traub: 学習は純電位依存 8 ゲート [M,S,N,C,A,H,R,B] (定数 TRAUB_V_LEAK 基準で
    # params 非依存)。Ca サブ系 XI/Q は extra(physics) へ分離し、dXI/i_ca が読む
    # {phi_area,g_Ca,V_Ca,Beta} は置換先ノード自身の p で解く。→ latent に params が
    # 一切焼込まれず compatible=True。1 サロゲートを traub19 全 comp へ (各々の
    # Ca params/area で) 移植可能。
    "traub": HybridPhysics(
        param_cls=TraubParams,
        dv=traub_dv,
        dv_cost=TRAUB_DV_COST,
        v_init=lambda p: p.V_LEAK,
        n_learned=len(TRAUB_LEARNED_GATE_NAMES),
        extra=ExtraPhysics(
            names=TRAUB_EXTRA_GATE_NAMES,
            step=traub_calcium_step,
            inits=traub_extra_inits,
            cost=TRAUB_CA_COST,
        ),
        compatible=lambda a, b: True,
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

    def _train_gate(self) -> np.ndarray:
        # 学習は先頭 n_learned ゲートのみ (extra=Ca サブ系は physics へ分離)。
        return access.gate_matrix(self._train_xr, self._meta.train_comp_id)[
            :, : self._physics.n_learned
        ]

    def fit(self, optimizer, library_specs: list[dict]) -> None:
        self._sindy_bundle = SINDyBundle.from_sindy(
            library_specs=library_specs,
            optimizer_spec=optimizer,
            x=self.preprocessor.encode(self._train_gate()),
            u=access.potential(self._train_xr, self._meta.train_comp_id),
            t=access.time(self._train_xr),
            targets=[sp.Symbol(v) for v in access.latent_vars(self._meta.n_components)],
            inputs=[sp.Symbol(access.POTENTIAL_VAR)],
            # 列構造: [g1..gN, V]。gate 群が先頭、末尾に V。u は入力に無し。
            roles=Roles(
                V=self._meta.n_components,
                g=list(range(self._meta.n_components)),
            ),
        )

    def params_compatible(self, comp: Compartment) -> bool:
        return self._physics.compatible(
            self.meta.train_comp.resolved_params,  # type: ignore[arg-type]
            comp.resolved_params,  # type: ignore[arg-type]
        )

    @property
    def surr_comp_type(self) -> CompartmentType:
        phys = self._physics
        extra = phys.extra
        xi = jnp.asarray(self.sindy_bundle.xi)
        decode = self.preprocessor.decode
        compute_theta = self.sindy_bundle.compute_theta()
        n_latent = self._meta.n_components

        # surr state = [latent₁..latentₙ, *extra]。extra は physics で解く追加状態
        # (Ca サブ系)、無ければ学習ゲート=dv 用の全ゲート。
        def hybrid_kernel(p, u_t, v, state):
            gates_learned = decode(state[:n_latent])
            if extra is None:
                gates, dextra = gates_learned, None
            else:
                gates, dextra = extra.step(p, v, gates_learned, state[n_latent:])
            dv = phys.dv(p, u_t, v, gates)
            dlatent = xi @ compute_theta(*state[:n_latent], v)
            return dv, dlatent if dextra is None else jnp.concatenate([dlatent, dextra])

        def node_inits(p) -> list[float]:
            # 置換先ノード自身の params で解く。latent は params-free ゲートの射影で
            # 全 comp 共通だが、静止電位 (V_LEAK/E_REST) と Ca サブ系 XI/Q
            # (phi_area/g_Ca 依存) はノードごとに違う。
            return (
                [phys.v_init(p)]
                + self.preprocessor.gate_inits
                + ([] if extra is None else extra.inits(p))
            )

        return CompartmentType(
            name="hybrid_surr",
            kernel=hybrid_kernel,
            param_cls=phys.param_cls,
            gate_names=access.latent_vars(n_latent)
            + ([] if extra is None else extra.names),
            inits=node_inits,
            opcost=None,
        )

    @property
    def opcost(self) -> OpCost:
        # kernel の 1 ステップ = decode(latent→gate) + Ca サブ系 physics (あれば) +
        # 物理 dV/dt + SINDy 潜在方程式。
        extra = self._physics.extra
        extra_cost = OpCost() if extra is None else extra.cost
        return (
            self.preprocessor.opcost()
            + extra_cost
            + self._physics.dv_cost
            + self.sindy_bundle.opcost()
        )
