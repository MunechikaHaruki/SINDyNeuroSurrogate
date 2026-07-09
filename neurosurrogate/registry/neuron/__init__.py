from collections import Counter

from ...core.network import Compartment, Edge, NeuronGraph
from ..compartments import COMPARTMENT_TYPES
from ..compartments.hh import HH_TYPE, PASSIVE_TYPE, HHParams
from ..compartments.traub import TRAUB_TYPE, TraubParams
from .traub19 import build_traub19


def chain(
    node_types: list[str],
    weights: list[float],
    stim: int = 0,
) -> NeuronGraph:
    """type 名リストから直鎖 NeuronGraph 構築。

    ノード名は型の頭文字 + 0始まり連番。
    例: ["passive","hh","passive"] → ["p0","h0","p1"]
    """
    expected = len(node_types) - 1
    assert len(weights) == expected, (
        f"weights の長さは len(node_types) - 1 = {expected} 必要"
    )
    counters: Counter = Counter()
    nodes = []
    for t in node_types:
        prefix = t[0]
        nodes.append(
            Compartment(name=f"{prefix}{counters[prefix]}", type=COMPARTMENT_TYPES[t])
        )
        counters[prefix] += 1
    return NeuronGraph(
        nodes=nodes,
        edges=[
            Edge(nodes[i].name, nodes[i + 1].name, w) for i, w in enumerate(weights)
        ],
        stim=nodes[stim].name,
    )


# per-compartment パラメータ例
# soma: デフォルト (高 G_NA)、dendrite: G_NA/G_K 低減で発火閾値↑
_HH_DENDRITE_PARAMS = HHParams(G_NA=60.0, G_K=18.0, G_LEAK=0.5)

# Traub: soma (デフォルト) vs dendrite (Na/K_DR/K_A 低減)
_TRAUB_DENDRITE_PARAMS = TraubParams(g_Na=5.0, g_K_DR=10.0, g_K_A=1.0)

MCMODELS: dict[str, NeuronGraph] = {
    "hh": NeuronGraph(
        nodes=[Compartment(name="soma", type=HH_TYPE)],
        edges=[],
        stim="soma",
    ),
    "traub": NeuronGraph(
        nodes=[Compartment(name="soma", type=TRAUB_TYPE)],
        edges=[],
        stim="soma",
    ),
    "php": chain(["passive", "hh", "passive"], [1.0, 0.7]),
    "hhp": chain(["hh", "hh", "passive"], [1.0, 0.7]),
    "pph": chain(["passive", "hh", "hh"], [1.0, 0.7]),
    "phhpp": chain(["passive", "hh", "hh", "passive", "passive"], [1.0, 0.7, 0.7, 0.5]),
    "pphhp": chain(["passive", "passive", "hh", "hh", "passive"], [1.0, 0.7, 0.7, 0.5]),
    "phhhp": chain(["passive", "hh", "hh", "hh", "passive"], [1.0, 0.7, 0.7, 0.5]),
    "hh_multi": NeuronGraph(
        nodes=[
            Compartment(name="soma", type=HH_TYPE),
            Compartment(name="d1", type=HH_TYPE, params=_HH_DENDRITE_PARAMS),
            Compartment(name="d2", type=HH_TYPE, params=_HH_DENDRITE_PARAMS),
        ],
        edges=[Edge("soma", "d1", 1.0), Edge("d1", "d2", 0.7)],
        stim="soma",
    ),
    "traub_multi": NeuronGraph(
        nodes=[
            Compartment(name="soma", type=TRAUB_TYPE),
            Compartment(name="d1", type=TRAUB_TYPE, params=_TRAUB_DENDRITE_PARAMS),
            Compartment(name="d2", type=TRAUB_TYPE, params=_TRAUB_DENDRITE_PARAMS),
        ],
        edges=[Edge("soma", "d1", 1.0), Edge("d1", "d2", 0.7)],
        stim="soma",
    ),
    "traub19": build_traub19(),
    "hh7": NeuronGraph(
        nodes=[
            Compartment(name="p1", type=PASSIVE_TYPE),
            Compartment(name="h1", type=HH_TYPE),
            Compartment(name="h2", type=HH_TYPE),
            Compartment(name="h3", type=HH_TYPE),
            Compartment(name="h4", type=HH_TYPE),
            Compartment(name="p2", type=PASSIVE_TYPE),
            Compartment(name="p3", type=PASSIVE_TYPE),
        ],
        edges=[
            Edge("p1", "h1", 1.0),
            Edge("h1", "h2", 0.7),
            Edge("h2", "h3", 0.7),
            Edge("h2", "h4", 0.5),
            Edge("h3", "p2", 0.5),
            Edge("h4", "p3", 0.6),
        ],
        stim="p1",
    ),
}
