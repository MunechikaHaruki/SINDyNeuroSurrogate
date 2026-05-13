from collections import Counter

from .model_dataset import Edge, NeuronGraph, Node


def chain(
    node_types: list[str],
    weights: list[float],
    stim: int = 0,
) -> NeuronGraph:
    """
    ノード名は型の頭文字 + 0始まり連番で自動生成
    例: ["passive", "hh", "passive"] → ["p0", "h0", "p1"]
    """
    assert len(weights) == len(node_types) - 1, (
        f"weights の長さは len(node_types) - 1 = {len(node_types) - 1} である必要があります"
    )
    counters: Counter = Counter()
    nodes = []
    for t in node_types:
        prefix = t[0]  # "hh" → "h", "passive" → "p"
        nodes.append(Node(f"{prefix}{counters[prefix]}", t))
        counters[prefix] += 1

    edges = [Edge(nodes[i].name, nodes[i + 1].name, w) for i, w in enumerate(weights)]
    return NeuronGraph(nodes=nodes, edges=edges, stim=nodes[stim].name)


MCMODELS: dict[str, NeuronGraph] = {
    "hh": NeuronGraph(
        nodes=[Node("soma", "hh")],
        edges=[],
        stim="soma",
    ),
    "php": chain(["passive", "hh", "passive"], [1.0, 0.7]),
    "hhp": chain(["hh", "hh", "passive"], [1.0, 0.7]),
    "pph": chain(["passive", "hh", "hh"], [1.0, 0.7]),
    "phhpp": chain(["passive", "hh", "hh", "passive", "passive"], [1.0, 0.7, 0.7, 0.5]),
    "pphhp": chain(["passive", "passive", "hh", "hh", "passive"], [1.0, 0.7, 0.7, 0.5]),
    "phhhp": chain(["passive", "hh", "hh", "hh", "passive"], [1.0, 0.7, 0.7, 0.5]),
    "hh7": NeuronGraph(
        nodes=[
            Node("p1", "passive"),
            Node("h1", "hh"),
            Node("h2", "hh"),
            Node("h3", "hh"),
            Node("h4", "hh"),
            Node("p2", "passive"),
            Node("p3", "passive"),
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
