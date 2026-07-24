"""Hybrid kernel の共有部品 (継承でなく関数合成で共有する leaf)。

物理 dV/dt + decode(latent) の骨格は SINDy 版 (hybrid.py) も UDE 版 (ude.py) も同一。
違うのは潜在方程式をどう同定し (各 ansatz の fit) どう評価するか (surr_comp_type が渡す
dlatent_fn) だけ — その2点だけ各 ansatz が持ち、共通骨格はここへ集約する。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import xarray as xr

from ....compartments.hh import HH_DV_COST, HHParams, hh_dv
from ....compartments.traub import (
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
from ....core import access
from ....core.network import CompartmentType
from ....core.opcost import OpCost
from ...closure.base import Closure
from ...meta import SurrogateMeta
from ...preprocessor.base import Preprocessor
from ..base import TrainInputs, TrainSource


@dataclass(frozen=True)
class ExtraPhysics:
    """学習 latent から外し physics で解く追加状態 (Traub の Ca サブ系 XI/Q)。params を
    陽に読む dynamics を学習に含めると params が latent へ焼込まれ置換に一致を要求する
    → physics 化で学習は純 params-free ゲートのみになる。

    names : latent の後へ付く状態名 (init 順)。
    step  : (p, v, gates_learned, extra_state) -> (dv 用の全ゲート, d(extra))。
    inits : (p) -> extra 初期値。
    cost  : extra step の演算コスト。
    """

    names: list[str]
    step: Callable
    inits: Callable[[Any], list[float]]
    cost: OpCost


@dataclass(frozen=True)
class HybridPhysics:
    """学習型ごとの物理 dV/dt 差分 (hybrid の dispatch 単位)。

    dv        : (params, u_t, v, gates) -> dv。
    dv_cost    : dv の演算コスト。
    v_init    : (params) -> 初期電位 (静止電位)。
    n_learned : 学習 latent が圧縮する先頭ゲート数。
    extra     : physics で解く追加状態 (None → 学習ゲート=全ゲート)。
    """

    param_cls: type
    dv: Callable
    dv_cost: OpCost
    v_init: Callable[[Any], float]
    n_learned: int
    extra: ExtraPhysics | None


# キー = meta.physics_type (既定 comp_type 名)。学習/physics の分割位置を preset で
# 振れる = アブレーション軸。
HYBRID_PHYSICS: dict[str, HybridPhysics] = {
    # HH: 3 ゲート全て純電位依存、Ca 無し → extra 無し。
    "hh": HybridPhysics(
        param_cls=HHParams,
        dv=hh_dv,
        dv_cost=HH_DV_COST,
        v_init=lambda p: p.E_REST,
        n_learned=3,
        extra=None,
    ),
    # Traub: 純電位依存 8 ゲート [M,S,N,C,A,H,R,B] を学習、Ca サブ系 XI/Q は extra へ
    # (params 依存を latent から締め出す → 1 サロゲートを traub19 全 comp へ移植可)。
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
    # Ca 電流ゲート S,R も physics へ (学習 6 ゲート)。i_ca=g_Ca·S²·R が XI を駆動し
    # decode 誤差が 2 乗で溜まる経路を断つ。
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


def hybrid_physics(meta: SurrogateMeta) -> HybridPhysics:
    """学習型に対応する物理差分 (既定は comp_type 名)。"""
    return HYBRID_PHYSICS[meta.physics_type or meta.comp_type.name]


def hybrid_train_inputs(
    source: TrainSource,
    train_xr: xr.Dataset,
    preprocessor: Preprocessor,
    n_components: int,
) -> TrainInputs:
    """状態は潜在のみ (V は物理式で解き入力で与える)。comp ごとに 1 軌道 (縦連結は
    偽微分)。"""
    return TrainInputs(
        x_names=access.latent_vars(n_components),
        u_names=[access.POTENTIAL_VAR],
        x=[preprocessor.encode(g) for g in source.gates(train_xr)],
        u=[v[:, None] for v in source.potentials(train_xr)],
    )


def hybrid_surr_comp_type(
    meta: SurrogateMeta,
    preprocessor: Preprocessor,
    closure: Closure,
    dlatent_fn: Callable,
) -> CompartmentType:
    """物理 dV/dt + decode(latent) + 潜在方程式 の kernel を組む。潜在方程式の評価
    `dlatent_fn: (latent, V) -> d(latent)/dt` だけ呼び手が渡す (SINDy=ξ 内積 / UDE=NN)。
    """
    phys = hybrid_physics(meta)
    extra = phys.extra
    decode = preprocessor.decode
    n_latent = meta.n_components

    # 1 ステップ = decode + Ca physics (あれば) + dV/dt + 潜在方程式。
    opcost = (
        preprocessor.opcost()
        + (OpCost() if extra is None else extra.cost)
        + phys.dv_cost
        + closure.opcost()
    )

    # surr state = [latent₁..ₙ, *extra]。extra 無ければ学習ゲート=全ゲート。
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
        # 静止電位と Ca サブ系はノードごと (params 依存)、latent は params-free で
        # 全 comp 共通。
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
        opcost=opcost,
    )
