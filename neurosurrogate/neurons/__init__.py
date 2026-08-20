"""**組み上がったニューロンモデルのカタログ** (`SimSpec.target` が名前で引く)。

作り方 (`_generate.py`) も per-comp 定数 (`traub19.py`) もこのディレクトリが持つので、
組んだ結果もここに置く。使う側 (`sim.spec`) の語彙ではなくニューロンの語彙。

**多 comp は traub19 系だけ**が生きている。以下にコメントアウトしてある chain 系
(`php`/`hhp`/…) と手組みの `hh_multi`/`traub_multi`/`hh7` は動作確認用に適当な
edge weight で繋いだだけのモデルで、面積 (comp ごとの area) も軸索 conductance も
実測値ではない = coupling が物理的に意味を持たない。単一 comp (`hh`/`traub`) は
coupling が無いので面積に依らず有効。復活させるなら per-comp の面積と g_axial を
与えてからにする。
"""

from ..core.network import Compartment, NeuronGraph
from ._generate import build_traub19
from .compartments.hh import HH_TYPE
from .compartments.traub import TRAUB_TYPE

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
    "traub19": build_traub19(),
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
