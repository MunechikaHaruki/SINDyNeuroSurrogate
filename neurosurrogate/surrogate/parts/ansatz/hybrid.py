"""物理 dV/dt と学習した潜在方程式を組み合わせる hybrid 定式化。"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import jax.numpy as jnp
import numpy as np
import xarray as xr

from ....core import access
from ....core.network import CompartmentType
from ....core.opcost import OpCost
from ....neurons import HYBRID_SPLITS, HybridSplit
from .. import Ansatz, Closure, Preprocessor
from ..closure.sindy import SINDyBundle
from ..closure.sindy.roles import Roles
from ..train_inputs import TrainInputs

if TYPE_CHECKING:
    from ...model import SurrogateSpec

C = TypeVar("C", bound=Closure)
_DLatent = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def _hybrid_split(*, physics_type: str) -> HybridSplit:
    """学習/physics 分割を選ぶ。**ansatz ブロックの唯一の宛先**で、他のキーを書けば
    TypeError。

    **既定値を持たない** = hybrid 系 preset は分割を必ず名指しする。comp_type 名で
    代替しない: `HYBRID_SPLITS` のキーは**分割**の名前で、comp 型の名前と一致して
    いるのは偶然にすぎず (同じ comp 型に分割が複数ある)、暗黙に落ちると分割を
    決める源が spec と ansatz の 2 つになる。
    """
    return HYBRID_SPLITS[physics_type]


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

    @classmethod
    def split(cls, spec: SurrogateSpec) -> HybridSplit:
        """この仕様の学習/physics 分割 (ansatz ブロックが名指しする。中身は neurons が
        持つ)。学習ゲート数も kernel の physics 側もここから出る。"""
        return _hybrid_split(**spec.ansatz_config)

    @classmethod
    def n_train_gate(cls, spec: SurrogateSpec) -> int:
        """純電位依存ゲートのみ学習し、Ca サブ系は physics へ分離する。"""
        return cls.split(spec).n_learned

    @classmethod
    def training_gates(
        cls, spec: SurrogateSpec, training_data: xr.Dataset
    ) -> list[np.ndarray]:
        """先頭 n_train_gate 本だけ切り出す (残りは physics 側が持つ)。"""
        return [
            gate[:, : cls.n_train_gate(spec)]
            for gate in access.gate_matrices(training_data, spec.train_comp_ids())
        ]

    @classmethod
    def train_inputs(
        cls,
        spec: SurrogateSpec,
        training_data: xr.Dataset,
        preprocessor: Preprocessor,
    ) -> TrainInputs:
        return TrainInputs(
            x_names=access.latent_vars(spec.n_components),
            u_names=[access.POTENTIAL_VAR],
            x=[preprocessor.encode(g) for g in cls.training_gates(spec, training_data)],
            u=[
                v[:, None]
                for v in access.potentials(training_data, spec.train_comp_ids())
            ],
        )

    @classmethod
    @abstractmethod
    def dlatent(cls, spec: SurrogateSpec, closure: C) -> _DLatent:
        """学習済み閉包項から ``(latent, V) -> dlatent/dt`` を作る。"""
        ...

    @classmethod
    def surr_comp_type(
        cls, spec: SurrogateSpec, preprocessor: Preprocessor, closure: C
    ) -> CompartmentType:
        split = cls.split(spec)
        extra = split.extra
        decode = preprocessor.decode
        dlatent = cls.dlatent(spec, closure)
        n_latent = spec.n_components
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
            name=spec.surr_type_name(),
            kernel=kernel,
            param_cls=split.param_cls,
            gate_names=access.latent_vars(n_latent)
            + ([] if extra is None else extra.names),
            inits=node_inits,
            opcost=opcost,
        )


class HybridSINDyAnsatz(HybridAnsatz[SINDyBundle]):
    """Hybrid の潜在方程式を SINDy で同定する。"""

    @classmethod
    def fit_closure(
        cls,
        spec: SurrogateSpec,
        training_data: xr.Dataset,
        preprocessor: Preprocessor,
    ) -> SINDyBundle:
        n = spec.n_components
        return SINDyBundle.from_sindy(
            cls.train_inputs(spec, training_data, preprocessor),
            access.time(training_data),
            Roles(V=n, g=list(range(n))),
            **spec.closure_config,
        )

    @classmethod
    def dlatent(cls, spec: SurrogateSpec, closure: SINDyBundle) -> _DLatent:
        xi = jnp.asarray(closure.xi)
        compute_theta = closure.compute_theta()
        return lambda latent, v: xi @ compute_theta(*latent, v)
