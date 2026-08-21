"""**NeuronGraph の語彙一式**: comp 型の対応表・組み方・組み上がったモデルのカタログ
(`SimSpec.target` が名前で引く)、および hybrid サロゲートの学習/physics 分割
(`HYBRID_SPLITS`)。

per-comp 定数 (`traub19.py`) も comp 型の実装 (`hh.py`/`traub.py`) もこのディレクトリが
持つので、それらを組んだ結果もここに置く。使う側 (`sim.spec` / `surrogate`) の語彙では
なくニューロンの語彙 — どのゲートが純電位依存でどれが Ca サブ系か、物理 dV/dt が何か
は、置換の定式化でなくモデルそのものが決める。

**多 comp は traub19 系だけ**が生きている。以下にコメントアウトしてある chain 系
(`php`/`hhp`/…) と手組みの `hh_multi`/`traub_multi`/`hh7` は動作確認用に適当な
edge weight で繋いだだけのモデルで、面積 (comp ごとの area) も軸索 conductance も
実測値ではない = coupling が物理的に意味を持たない。単一 comp (`hh`/`traub`) は
coupling が無いので面積に依らず有効。復活させるなら per-comp の面積と g_axial を
与えてからにする。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.network import Compartment, Edge, NeuronGraph
from ..core.opcost import OpCost
from .hh import HH_DV_COST, HH_TYPE, PASSIVE_TYPE, HHParams, hh_dv
from .traub import (
    TRAUB_CA_COST,
    TRAUB_DV_COST,
    TRAUB_EXTRA_GATE_NAMES,
    TRAUB_LEARNED_GATE_NAMES,
    TRAUB_SR_CA_COST,
    TRAUB_SR_EXTRA_GATE_NAMES,
    TRAUB_SR_LEARNED_GATE_NAMES,
    TRAUB_TYPE,
    TraubParams,
    traub_calcium_step,
    traub_dv,
    traub_extra_inits,
    traub_sr_calcium_step,
    traub_sr_extra_inits,
)
from .traub19 import NC, g_axial, name_at, params_at

# type 名文字列 → CompartmentType の dispatch table (from_dict 等で使用)
COMPARTMENT_TYPES = {
    "hh": HH_TYPE,
    "passive": PASSIVE_TYPE,
    "traub": TRAUB_TYPE,
}


@dataclass(frozen=True)
class _ExtraPhysics:
    """hybrid で学習 latent から外し、physics で解き続ける追加状態。"""

    names: list[str]
    step: Callable
    inits: Callable[[Any], list[float]]
    cost: OpCost


@dataclass(frozen=True)
class HybridSplit:
    """hybrid サロゲートの「どこまでを学習し、どこからを物理式で解くか」。

    分割位置 (`n_learned`) も残す物理 (`dv` / `extra`) も**ニューロンモデルの性質**
    (状態の並び順とイオン電流の構造で決まる) なので、定式化側でなくここが持つ。
    サロゲートはこれを読んで kernel を組むだけで、どのゲートが何かを知らない。
    """

    param_cls: type
    dv: Callable
    dv_cost: OpCost
    v_init: Callable[[Any], float]
    n_learned: int
    extra: _ExtraPhysics | None


# キー = preset の ansatz.physics_type (必須・既定なし)。同じ comp 型に対して
# 分割位置の違う版を並べられる = preset が必ず名指しする。
HYBRID_SPLITS: dict[str, HybridSplit] = {
    "hh": HybridSplit(
        param_cls=HHParams,
        dv=hh_dv,
        dv_cost=HH_DV_COST,
        v_init=lambda p: p.E_REST,
        n_learned=3,
        extra=None,
    ),
    "traub": HybridSplit(
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
    "traub_sr_physics": HybridSplit(
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


def _build_traub19() -> NeuronGraph:
    """19-comp Traub モデルを組む (per-comp 定数は `traub19.py`)。

    全 comp 同一 traub 型。注入ノードは形態でなく `SimSpec.stim` が、置換範囲は
    `EvalSeries.replace_targets` が決める = **形態はどちらも知らない**。
    """
    nodes = [
        Compartment(name=name_at(i), type=TRAUB_TYPE, params=params_at(i))
        for i in range(NC)
    ]
    edges = [Edge(name_at(i), name_at(i + 1), g_axial(i)) for i in range(NC - 1)]
    return NeuronGraph(nodes=nodes, edges=edges)


MCMODELS: dict[str, NeuronGraph] = {
    "hh": NeuronGraph(
        nodes=[Compartment(name="soma", type=HH_TYPE)],
        edges=[],
    ),
    "traub": NeuronGraph(
        nodes=[Compartment(name="soma", type=TRAUB_TYPE)],
        edges=[],
    ),
    # 全 comp 同一 traub 型 (元の traub.c と同じ soma 注入)。どこを置換するかは
    # 形態でなく実験の記述が決める (`EvalSeries.replace_targets`) ので、置換範囲を
    # 絞るためだけの双子モデルは持たない。
    "traub19": _build_traub19(),
}

# 過去モデル

# def chain(node_types: list[str], weights: list[float]) -> NeuronGraph:
#     """type 名リストから直鎖 NeuronGraph 構築。
#
#     ノード名は型の頭文字 + 0始まり連番。ただし最初の非 passive (=細胞体) は "soma"
#     と命名し全モデル共通の soma 規約に揃える (train_comp_identifier 既定と一致)。
#     例: ["passive","hh","passive"] → ["p0","soma","p1"]
#     """
#     expected = len(node_types) - 1
#     assert len(weights) == expected, (
#         f"weights の長さは len(node_types) - 1 = {expected} 必要"
#     )
#     soma_idx = next(
#         (i for i, t in enumerate(node_types) if t != "passive"),
#         None,
#     )
#     assert soma_idx is not None, "chain に非 passive (soma) ノードが必要"
#     counters: Counter = Counter()
#     nodes = []
#     for i, t in enumerate(node_types):
#         if i == soma_idx:
#             name = "soma"
#         else:
#             name = f"{t[0]}{counters[t[0]]}"
#             counters[t[0]] += 1
#         nodes.append(Compartment(name=name, type=COMPARTMENT_TYPES[t]))
#     return NeuronGraph(
#         nodes=nodes,
#         edges=[
#             Edge(nodes[i].name, nodes[i + 1].name, w) for i, w in enumerate(weights)
#         ],
#     )

# --- 面積・軸索 conductance が未考慮の暫定モデル群 (上の docstring 参照) ---
#
# # per-compartment パラメータ例
# # soma: デフォルト (高 G_NA)、dendrite: G_NA/G_K 低減で発火閾値↑
# _HH_DENDRITE_PARAMS = HHParams(G_NA=60.0, G_K=18.0, G_LEAK=0.5)
#
# # Traub: soma (デフォルト) vs dendrite (Na/K_DR/K_A 低減)
# _TRAUB_DENDRITE_PARAMS = TraubParams(g_Na=5.0, g_K_DR=10.0, g_K_A=1.0)
#
# MCMODELS |= {
#     "php": chain(["passive", "hh", "passive"], [1.0, 0.7]),
#     "hhp": chain(["hh", "hh", "passive"], [1.0, 0.7]),
#     "pph": chain(["passive", "hh", "hh"], [1.0, 0.7]),
#     "phhpp": chain(
#         ["passive", "hh", "hh", "passive", "passive"], [1.0, 0.7, 0.7, 0.5]
#     ),
#     "pphhp": chain(
#         ["passive", "passive", "hh", "hh", "passive"], [1.0, 0.7, 0.7, 0.5]
#     ),
#     "phhhp": chain(["passive", "hh", "hh", "hh", "passive"], [1.0, 0.7, 0.7, 0.5]),
#     "hh_multi": NeuronGraph(
#         nodes=[
#             Compartment(name="soma", type=HH_TYPE),
#             Compartment(name="d1", type=HH_TYPE, params=_HH_DENDRITE_PARAMS),
#             Compartment(name="d2", type=HH_TYPE, params=_HH_DENDRITE_PARAMS),
#         ],
#         edges=[Edge("soma", "d1", 1.0), Edge("d1", "d2", 0.7)],
#     ),
#     "traub_multi": NeuronGraph(
#         nodes=[
#             Compartment(name="soma", type=TRAUB_TYPE),
#             Compartment(name="d1", type=TRAUB_TYPE, params=_TRAUB_DENDRITE_PARAMS),
#             Compartment(name="d2", type=TRAUB_TYPE, params=_TRAUB_DENDRITE_PARAMS),
#         ],
#         edges=[Edge("soma", "d1", 1.0), Edge("d1", "d2", 0.7)],
#     ),
#     "hh7": NeuronGraph(
#         nodes=[
#             Compartment(name="p1", type=PASSIVE_TYPE),
#             Compartment(name="soma", type=HH_TYPE),
#             Compartment(name="h2", type=HH_TYPE),
#             Compartment(name="h3", type=HH_TYPE),
#             Compartment(name="h4", type=HH_TYPE),
#             Compartment(name="p2", type=PASSIVE_TYPE),
#             Compartment(name="p3", type=PASSIVE_TYPE),
#         ],
#         edges=[
#             Edge("p1", "soma", 1.0),
#             Edge("soma", "h2", 0.7),
#             Edge("h2", "h3", 0.7),
#             Edge("h2", "h4", 0.5),
#             Edge("h3", "p2", 0.5),
#             Edge("h4", "p3", 0.6),
#         ],
#     ),
# }
