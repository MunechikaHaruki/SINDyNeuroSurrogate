"""物理 dV/dt と学習した潜在方程式を組み合わせる hybrid 定式化。"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TypeVar

import jax.numpy as jnp
import xarray as xr

from ....core import access
from ....core.network import CompartmentType
from ....core.opcost import OpCost
from ....neurons import HYBRID_SPLITS, HybridSplit
from .. import Ansatz, Closure, Preprocessor, TrainInputs
from ..closure.sindy import SINDyBundle
from ..closure.sindy.roles import Roles

C = TypeVar("C", bound=Closure)
_DLatent = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


class HybridAnsatz(Ansatz[C]):
    """Hybrid 共通骨格。

    サブクラスの差分は閉包項の同定 ``fit_closure`` と、その閉包項から潜在方程式を作る
    ``dlatent`` だけ。学習列と kernel の組立順序はこの interface の内側に置く。
    """

    @classmethod
    def params_match(cls, train: tuple | None, node: tuple | None) -> bool:
        """回路 params を問わず適用できる: physics 側が params を**入力として**受け、
        閉包項は潜在ゲートの形だけを担うので係数に params が焼き込まれない。"""
        return True

    @property
    def _split(self) -> HybridSplit:
        """この仕様の学習/physics 分割 (spec が名前で選ぶ。中身は neurons が持つ)。"""
        return HYBRID_SPLITS[self.spec.physics_type or self.spec.comp_type.name]

    def n_train_gate(self) -> int:
        """純電位依存ゲートのみ学習し、Ca サブ系は physics へ分離する。"""
        return self._split.n_learned

    def train_inputs(
        self, training_data: xr.Dataset, preprocessor: Preprocessor
    ) -> TrainInputs:
        return TrainInputs(
            x_names=access.latent_vars(self.spec.n_components),
            u_names=[access.POTENTIAL_VAR],
            x=[preprocessor.encode(g) for g in self.training_gates(training_data)],
            u=[
                v[:, None]
                for v in access.potentials(training_data, self.spec.train_comp_ids())
            ],
        )

    @abstractmethod
    def dlatent(self, preprocessor: Preprocessor, closure: C) -> _DLatent:
        """学習済み閉包項から ``(latent, V) -> dlatent/dt`` を作る。"""
        ...

    def surr_comp_type(self, preprocessor: Preprocessor, closure: C) -> CompartmentType:
        split = self._split
        extra = split.extra
        decode = preprocessor.decode
        dlatent = self.dlatent(preprocessor, closure)
        n_latent = self.spec.n_components
        opcost = (
            preprocessor.opcost()
            + (OpCost() if extra is None else extra.cost)
            + split.dv_cost
            + closure.opcost()
        )

        def kernel(p, u_t, v, state):
            gates_learned = decode(state[:n_latent])
            if extra is None:
                gates, dextra = gates_learned, None
            else:
                gates, dextra = extra.step(p, v, gates_learned, state[n_latent:])
            dv = split.dv(p, u_t, v, gates)
            dlatent_dt = dlatent(state[:n_latent], v)
            return (
                dv,
                dlatent_dt if dextra is None else jnp.concatenate([dlatent_dt, dextra]),
            )

        def node_inits(p) -> list[float]:
            return (
                [split.v_init(p)]
                + preprocessor.gate_inits
                + ([] if extra is None else extra.inits(p))
            )

        return CompartmentType(
            name=self.spec.surr_type_name(),
            kernel=kernel,
            param_cls=split.param_cls,
            gate_names=access.latent_vars(n_latent)
            + ([] if extra is None else extra.names),
            inits=node_inits,
            opcost=opcost,
        )


class HybridSINDyAnsatz(HybridAnsatz[SINDyBundle]):
    """Hybrid の潜在方程式を SINDy で同定する。"""

    def fit_closure(
        self, training_data: xr.Dataset, preprocessor: Preprocessor, config: dict
    ) -> SINDyBundle:
        n = self.spec.n_components
        return SINDyBundle.from_sindy(
            self.train_inputs(training_data, preprocessor),
            access.time(training_data),
            Roles(V=n, g=list(range(n))),
            config,
        )

    def dlatent(self, preprocessor: Preprocessor, closure: SINDyBundle) -> _DLatent:
        xi = jnp.asarray(closure.xi)
        compute_theta = closure.compute_theta()
        return lambda latent, v: xi @ compute_theta(*latent, v)
