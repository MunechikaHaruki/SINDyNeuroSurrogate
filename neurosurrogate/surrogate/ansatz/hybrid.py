from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import sympy as sp
import xarray as xr

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
from ...core.network import CompartmentType
from ...core.opcost import OpCost
from ..closure.sindy import SINDyBundle
from ..closure.sindy.roles import Roles
from ..meta import SurrogateMeta
from ..preprocessor.base import Preprocessor
from ..replace import replaceable
from .base import Ansatz


@dataclass(frozen=True)
class ExtraPhysics:
    """学習 latent から外し physics で解く追加状態 (Traub の Ca サブ系 XI/Q)。

    これらの dynamics は params を陽に読む (置換先ノード自身の p で解く) → 学習には
    含めない。含めれば params が latent へ焼込まれ、置換に params 一致を要求すること
    になる。physics 化することで学習は純 params-free ゲートのみになる。

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
    """

    param_cls: type
    dv: Callable
    dv_cost: OpCost
    v_init: Callable[[Any], float]
    n_learned: int
    extra: ExtraPhysics | None


HYBRID_PHYSICS: dict[str, HybridPhysics] = {
    # HH: 3 ゲート全てが純電位依存 (Ca 無し) → extra 無し。G_*/E_*/C は dv が陽に読む
    # → 置換先ノード自身の params で解ける。
    "hh": HybridPhysics(
        param_cls=HHParams,
        dv=hh_dv,
        dv_cost=HH_DV_COST,
        v_init=lambda p: p.E_REST,
        n_learned=3,
        extra=None,
    ),
    # Traub: 学習は純電位依存 8 ゲート [M,S,N,C,A,H,R,B] (定数 TRAUB_V_LEAK 基準で
    # params 非依存)。Ca サブ系 XI/Q は extra(physics) へ分離し、dXI/i_ca が読む
    # {phi_area,g_Ca,V_Ca,Beta} は置換先ノード自身の p で解く。→ latent に params が
    # 一切焼込まれない。1 サロゲートを traub19 全 comp へ (各々の Ca params/area で)
    # 移植可能。
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
    ),
}


class HybridAnsatz(Ansatz[SINDyBundle]):
    """Hybrid: 物理回路方程式 (params 陽) + 学習ゲート dynamics。
      dV/dt   = 学習型の物理式 (HYBRID_PHYSICS[train_type].dv)
      gates   = decode(latent)                        (線形/AE decoder)
      d(latent)/dt = f(V, latent) via SINDy
    SINDy 入力順: (g1, ..., V) → library_specs は index 0=latent, 末尾=V。
    HH/Traub 両対応 (gate 数・param_cls・dv は学習型から解決)。

    潜在方程式 f をどう表現するかは `_dlatent` の 1 点に閉じている (kernel の骨格・
    physics・初期値は f の表現に依らない)。f を NN + ODE 解経由の学習に替えた
    もの = UDE で、その差分は `fit`/`_dlatent`/`opcost` の 3 本だけになる。
    """

    def _physics(self, meta: SurrogateMeta) -> HybridPhysics:
        return HYBRID_PHYSICS[meta.train_comp_type.name]

    def _train_comp_ids(self, meta: SurrogateMeta) -> list[int]:
        """学習データ源の comp id 列 = 置換対象になるノード全部 (comp_id 昇順)。

        「置換する comp の軌道で学習する」ことで訓練分布を評価分布に一致させる。
        単体モデル (MCMODELS["traub"] 等) なら train_comp 1 個で従来と同一。学習
        データセットに multi-comp を指定すれば全 comp の (V, gate) 軌道が入る。
        学習ゲートは params-free (V 依存のみ) → 全 comp は同一のゲート多様体上に
        乗り、増えるのは V の被覆だけ。n_components を増やす必要はない。
        """
        return [
            i
            for i, comp in enumerate(meta.dataset.net.nodes)
            if replaceable(meta, comp)
        ]

    def _comp_gate(
        self, meta: SurrogateMeta, train_xr: xr.Dataset, comp_id: int
    ) -> np.ndarray:
        # 学習は先頭 n_learned ゲートのみ (extra=Ca サブ系は physics へ分離)。
        return access.gate_matrix(train_xr, comp_id)[:, : self._physics(meta).n_learned]

    def train_gate(self, meta: SurrogateMeta, train_xr: xr.Dataset) -> np.ndarray:
        # preprocessor は時間微分を取らない → 全 comp を縦連結してよい。gate_inits
        # は先頭行 = 最小 comp_id の t=0 潜在だが、初期ゲートは V_LEAK 定常で解かれ
        # params-free → 全 comp 同値。
        return np.concatenate(
            [self._comp_gate(meta, train_xr, i) for i in self._train_comp_ids(meta)],
            axis=0,
        )

    def fit(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
        spec: dict,
    ) -> SINDyBundle:
        comp_ids = self._train_comp_ids(meta)
        return SINDyBundle.from_sindy(
            library_specs=spec["library_specs"],
            optimizer_spec=spec["optimizer"],
            # comp ごとに 1 軌道のリストで渡す。SINDy は t から時間微分を数値推定する
            # ため、train_gate のように縦連結すると境界に偽の微分が入る。
            x=[
                preprocessor.encode(self._comp_gate(meta, train_xr, i))
                for i in comp_ids
            ],
            u=[access.potential(train_xr, i)[:, None] for i in comp_ids],
            t=[access.time(train_xr)] * len(comp_ids),
            targets=[sp.Symbol(v) for v in access.latent_vars(meta.n_components)],
            inputs=[sp.Symbol(access.POTENTIAL_VAR)],
            # 列構造: [g1..gN, V]。gate 群が先頭、末尾に V。u は入力に無し。
            roles=Roles(
                V=meta.n_components,
                g=list(range(meta.n_components)),
            ),
        )

    def _dlatent(self, closure: SINDyBundle) -> Callable:
        """潜在方程式の右辺 `(latent, V) -> d(latent)/dt`。

        **閉包項をどう評価するかを知る唯一の場所**。SINDy はライブラリ展開との
        内積、UDE なら NN 呼び出しに替わる。kernel 側はこの関数しか見ない。
        """
        xi = jnp.asarray(closure.xi)
        compute_theta = closure.compute_theta()
        return lambda latent, v: xi @ compute_theta(*latent, v)

    def surr_comp_type(
        self,
        meta: SurrogateMeta,
        preprocessor: Preprocessor,
        closure: SINDyBundle,
    ) -> CompartmentType:
        phys = self._physics(meta)
        extra = phys.extra
        decode = preprocessor.decode
        dlatent_fn = self._dlatent(closure)
        n_latent = meta.n_components

        # surr state = [latent₁..latentₙ, *extra]。extra は physics で解く追加状態
        # (Ca サブ系)、無ければ学習ゲート=dv 用の全ゲート。
        def hybrid_kernel(p, u_t, v, state):
            gates_learned = decode(state[:n_latent])
            if extra is None:
                gates, dextra = gates_learned, None
            else:
                gates, dextra = extra.step(p, v, gates_learned, state[n_latent:])
            dv = phys.dv(p, u_t, v, gates)
            dlatent = dlatent_fn(state[:n_latent], v)
            return dv, dlatent if dextra is None else jnp.concatenate([dlatent, dextra])

        def node_inits(p) -> list[float]:
            # 置換先ノード自身の params で解く。latent は params-free ゲートの射影で
            # 全 comp 共通だが、静止電位 (V_LEAK/E_REST) と Ca サブ系 XI/Q
            # (phi_area/g_Ca 依存) はノードごとに違う。
            return (
                [phys.v_init(p)]
                + preprocessor.gate_inits
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

    def opcost(
        self,
        meta: SurrogateMeta,
        preprocessor: Preprocessor,
        closure: SINDyBundle,
    ) -> OpCost:
        # kernel の 1 ステップ = decode(latent→gate) + Ca サブ系 physics (あれば) +
        # 物理 dV/dt + 潜在方程式。
        phys = self._physics(meta)
        return (
            preprocessor.opcost()
            + (OpCost() if phys.extra is None else phys.extra.cost)
            + phys.dv_cost
            + closure.opcost()
        )
