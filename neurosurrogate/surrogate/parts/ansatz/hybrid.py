"""物理 dV/dt と学習した潜在方程式を組み合わせる hybrid 定式化。"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import jax.numpy as jnp
import xarray as xr

from ....core import access
from ....core.network import CompartmentType
from ....core.opcost import OpCost
from ....neurons.hh import HH_DV_COST, HHParams, hh_dv
from ....neurons.traub import (
    TRAUB_CA_COST,
    TRAUB_DV_COST,
    TRAUB_EXTRA_GATE_NAMES,
    TRAUB_LEARNED_GATE_NAMES,
    TRAUB_SR_CA_COST,
    TRAUB_SR_EXTRA_GATE_NAMES,
    TRAUB_SR_LEARNED_GATE_NAMES,
    TraubParams,
    traub_calcium_step,
    traub_dv,
    traub_extra_inits,
    traub_sr_calcium_step,
    traub_sr_extra_inits,
)
from .. import Ansatz, Closure, Preprocessor, TrainInputs
from ..closure.sindy import SINDyBundle
from ..closure.sindy.roles import Roles
from ._sindy_fit import fit_sindy

C = TypeVar("C", bound=Closure)
_DLatent = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

if TYPE_CHECKING:
    from ...model import SurrogateSpec


@dataclass(frozen=True)
class _ExtraPhysics:
    """学習 latent から外し physics で解く追加状態。"""

    names: list[str]
    step: Callable
    inits: Callable[[Any], list[float]]
    cost: OpCost


@dataclass(frozen=True)
class _HybridPhysics:
    """学習型ごとの物理 dV/dt と学習・physics の分割。"""

    param_cls: type
    dv: Callable
    dv_cost: OpCost
    v_init: Callable[[Any], float]
    n_learned: int
    extra: _ExtraPhysics | None


# キー = spec.physics_type (既定 comp_type 名)。
# 学習/physics の分割位置を preset で振る。
HYBRID_PHYSICS: dict[str, _HybridPhysics] = {
    "hh": _HybridPhysics(
        param_cls=HHParams,
        dv=hh_dv,
        dv_cost=HH_DV_COST,
        v_init=lambda p: p.E_REST,
        n_learned=3,
        extra=None,
    ),
    "traub": _HybridPhysics(
        param_cls=TraubParams,
        dv=traub_dv,
        dv_cost=TRAUB_DV_COST,
        v_init=lambda p: p.V_LEAK,
        n_learned=len(TRAUB_LEARNED_GATE_NAMES),
        extra=_ExtraPhysics(
            names=TRAUB_EXTRA_GATE_NAMES,
            step=traub_calcium_step,
            inits=traub_extra_inits,
            cost=TRAUB_CA_COST,
        ),
    ),
    "traub_sr_physics": _HybridPhysics(
        param_cls=TraubParams,
        dv=traub_dv,
        dv_cost=TRAUB_DV_COST,
        v_init=lambda p: p.V_LEAK,
        n_learned=len(TRAUB_SR_LEARNED_GATE_NAMES),
        extra=_ExtraPhysics(
            names=TRAUB_SR_EXTRA_GATE_NAMES,
            step=traub_sr_calcium_step,
            inits=traub_sr_extra_inits,
            cost=TRAUB_SR_CA_COST,
        ),
    ),
}


class HybridAnsatz(Ansatz[C]):
    """Hybrid 共通骨格。

    サブクラスの差分は閉包項の同定 ``fit`` と、その閉包項から潜在方程式を作る
    ``dlatent`` だけ。学習列と kernel の組立順序はこの interface の内側に置く。
    """

    def params_match(self, train: tuple | None, node: tuple | None) -> bool:
        """回路 params を問わず適用できる: physics 側が params を**入力として**受け、
        閉包項は潜在ゲートの形だけを担うので係数に params が焼き込まれない。"""
        return True

    def _physics(self, spec: SurrogateSpec) -> _HybridPhysics:
        return HYBRID_PHYSICS[spec.physics_type or spec.comp_type.name]

    def n_train_gate(self, spec: SurrogateSpec) -> int:
        """純電位依存ゲートのみ学習し、Ca サブ系は physics へ分離する。"""
        return self._physics(spec).n_learned

    def train_inputs(
        self,
        spec: SurrogateSpec,
        training_data: xr.Dataset,
        preprocessor: Preprocessor,
    ) -> TrainInputs:
        comp_ids = spec.train_comp_ids()
        return TrainInputs(
            x_names=access.latent_vars(spec.n_components),
            u_names=[access.POTENTIAL_VAR],
            x=[
                preprocessor.encode(g) for g in self.training_gates(spec, training_data)
            ],
            u=[v[:, None] for v in access.potentials(training_data, comp_ids)],
        )

    @abstractmethod
    def fit(
        self,
        spec: SurrogateSpec,
        training_data: xr.Dataset,
        preprocessor: Preprocessor,
        config: dict,
    ) -> C:
        """Hybrid の閉包項を同定する。"""
        ...

    @abstractmethod
    def dlatent(
        self,
        spec: SurrogateSpec,
        preprocessor: Preprocessor,
        closure: C,
    ) -> _DLatent:
        """学習済み閉包項から ``(latent, V) -> dlatent/dt`` を作る。"""
        ...

    def surr_comp_type(
        self,
        spec: SurrogateSpec,
        preprocessor: Preprocessor,
        closure: C,
    ) -> CompartmentType:
        phys = self._physics(spec)
        extra = phys.extra
        decode = preprocessor.decode
        dlatent = self.dlatent(spec, preprocessor, closure)
        n_latent = spec.n_components
        opcost = (
            preprocessor.opcost()
            + (OpCost() if extra is None else extra.cost)
            + phys.dv_cost
            + closure.opcost()
        )

        def kernel(p, u_t, v, state):
            gates_learned = decode(state[:n_latent])
            if extra is None:
                gates, dextra = gates_learned, None
            else:
                gates, dextra = extra.step(p, v, gates_learned, state[n_latent:])
            dv = phys.dv(p, u_t, v, gates)
            dlatent_dt = dlatent(state[:n_latent], v)
            return (
                dv,
                dlatent_dt if dextra is None else jnp.concatenate([dlatent_dt, dextra]),
            )

        def node_inits(p) -> list[float]:
            return (
                [phys.v_init(p)]
                + preprocessor.gate_inits
                + ([] if extra is None else extra.inits(p))
            )

        return CompartmentType(
            name=spec.surr_type_name(),
            kernel=kernel,
            param_cls=phys.param_cls,
            gate_names=access.latent_vars(n_latent)
            + ([] if extra is None else extra.names),
            inits=node_inits,
            opcost=opcost,
        )


class HybridSINDyAnsatz(HybridAnsatz[SINDyBundle]):
    """Hybrid の潜在方程式を SINDy で同定する。"""

    def fit(
        self,
        spec: SurrogateSpec,
        training_data: xr.Dataset,
        preprocessor: Preprocessor,
        config: dict,
    ) -> SINDyBundle:
        inputs = self.train_inputs(spec, training_data, preprocessor)
        roles = Roles(V=spec.n_components, g=list(range(spec.n_components)))
        return fit_sindy(inputs, access.time(training_data), roles, config)

    def dlatent(
        self,
        spec: SurrogateSpec,
        preprocessor: Preprocessor,
        closure: SINDyBundle,
    ) -> _DLatent:
        xi = jnp.asarray(closure.xi)
        compute_theta = closure.compute_theta()
        return lambda latent, v: xi @ compute_theta(*latent, v)
