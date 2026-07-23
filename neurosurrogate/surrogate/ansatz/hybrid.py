from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic

import jax.numpy as jnp
import sympy as sp
import xarray as xr

from ...compartments.hh import HH_DV_COST, HHParams, hh_dv
from ...compartments.traub import (
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
from ...core import access
from ...core.network import CompartmentType
from ...core.opcost import OpCost
from ..closure.sindy import SINDyBundle
from ..closure.sindy.roles import Roles
from ..meta import SurrogateMeta
from ..preprocessor.base import Preprocessor
from .base import Ansatz, C, TrainInputs


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


# キー = meta.physics_type (既定は comp_type 名)。同じ置換対象でも「どこまでを学習
# ゲートにし、どこから physics で解くか」を preset で振れる = アブレーションの軸。
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
    # Ca 電流のゲート S,R も physics へ回す (学習は 6 ゲート)。i_ca = g_Ca·S²·R は
    # XI の積分器を駆動するので、decode 誤差が 2 乗で増幅されバイアスとして溜まる
    # 経路を断つ狙い。"traub" との差は学習/physics の分割位置だけ。
    "traub_sr_physics": HybridPhysics(
        param_cls=TraubParams,
        dv=traub_dv,
        dv_cost=TRAUB_DV_COST,
        v_init=lambda p: p.V_LEAK,
        n_learned=len(TRAUB_SR_LEARNED_GATE_NAMES),
        extra=ExtraPhysics(
            names=TRAUB_SR_EXTRA_GATE_NAMES,
            step=traub_sr_calcium_step,
            inits=traub_sr_extra_inits,
            cost=TRAUB_SR_CA_COST,
        ),
    ),
}


class HybridBase(Ansatz[C], Generic[C]):
    """Hybrid: 物理回路方程式 (params 陽) + 学習ゲート dynamics。
      dV/dt   = 学習型の物理式 (HYBRID_PHYSICS[train_type].dv)
      gates   = decode(latent)                        (線形/AE decoder)
      d(latent)/dt = f(V, latent)                     (閉包項の表現は派生が決める)
    HH/Traub 両対応 (gate 数・param_cls・dv は学習型から解決)。

    **潜在方程式 f の表現に依らない部分をここに置く** — kernel の骨格・physics 分離・
    初期値・学習入力の列構造は f が ξ でも NN でも同じ。f がどう評価されるかを知る
    のは `_dlatent` の 1 点だけで、派生が実装するのは f 固有の 3 本
    (`fit` / `_dlatent` / `_closure_opcost`) に限られる。
      `HybridAnsatz` : f = SINDy ライブラリ項の疎な線形結合 (前処理と別々に学習)
      `UDEAnsatz`    : f = NN、encoder/decoder ごと ODE 解を通して joint 学習
    """

    def _physics(self, meta: SurrogateMeta) -> HybridPhysics:
        return HYBRID_PHYSICS[meta.physics_type or meta.comp_type.name]

    def n_train_gate(self, meta: SurrogateMeta) -> int:
        """純電位依存ゲートのみ学習 (extra=Ca サブ系は physics へ分離)。"""
        return self._physics(meta).n_learned

    def train_inputs(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
    ) -> TrainInputs:
        # 状態は潜在のみ (V は物理式で解くので同定せず、入力として与える)。comp ごと
        # に 1 軌道で分ける — SINDy は t から時間微分を数値推定するため、stacked_gate
        # のように縦連結すると境界に偽の微分が入る (UDE も窓を跨がせない)。
        source = self.train_source(meta)
        return TrainInputs(
            x_names=access.latent_vars(meta.n_components),
            u_names=[access.POTENTIAL_VAR],
            x=[preprocessor.encode(source.gate(train_xr, i)) for i in source.comp_ids],
            u=[access.potential(train_xr, i)[:, None] for i in source.comp_ids],
        )

    @abstractmethod
    def _dlatent(self, closure: C) -> Callable:
        """潜在方程式の右辺 `(latent, V) -> d(latent)/dt`。

        **閉包項をどう評価するかを知る唯一の場所**。SINDy はライブラリ展開との
        内積、UDE は NN 呼び出し。kernel 側はこの関数しか見ない。
        """
        ...

    @abstractmethod
    def _closure_opcost(self, closure: C) -> OpCost:
        """閉包項 1 回の評価コスト。`Closure` の契約には載せない (型を知らない
        呼び出し側が要求しない) ので、具体型を知る ansatz 側で引き出す。"""
        ...

    def surr_comp_type(
        self,
        meta: SurrogateMeta,
        preprocessor: Preprocessor,
        closure: C,
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
            name=meta.surr_type_name,
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
        closure: C,
    ) -> OpCost:
        # kernel の 1 ステップ = decode(latent→gate) + Ca サブ系 physics (あれば) +
        # 物理 dV/dt + 潜在方程式。閉包項の分だけが表現ごとに違う。
        phys = self._physics(meta)
        return (
            preprocessor.opcost()
            + (OpCost() if phys.extra is None else phys.extra.cost)
            + phys.dv_cost
            + self._closure_opcost(closure)
        )


class HybridAnsatz(HybridBase[SINDyBundle]):
    """潜在方程式を SINDy で同定する hybrid (前処理と閉包項を別々に学習する)。

    SINDy 入力順: (g1, ..., V) → library_specs は index 0=latent, 末尾=V。
    """

    def fit(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
        spec: dict,
    ) -> SINDyBundle:
        inputs = self.train_inputs(meta, train_xr, preprocessor)
        return SINDyBundle.from_sindy(
            library_specs=spec["library_specs"],
            optimizer_spec=spec["optimizer"],
            x=inputs.x,
            u=inputs.u,
            t=[access.time(train_xr)] * len(inputs.x),
            targets=[sp.Symbol(v) for v in inputs.x_names],
            inputs=[sp.Symbol(v) for v in inputs.u_names],
            # 列構造: [g1..gN, V]。gate 群が先頭、末尾に V。u は入力に無し。
            roles=Roles(
                V=meta.n_components,
                g=list(range(meta.n_components)),
            ),
        )

    def _dlatent(self, closure: SINDyBundle) -> Callable:
        xi = jnp.asarray(closure.xi)
        compute_theta = closure.compute_theta()
        return lambda latent, v: xi @ compute_theta(*latent, v)

    def _closure_opcost(self, closure: SINDyBundle) -> OpCost:
        return closure.opcost()
