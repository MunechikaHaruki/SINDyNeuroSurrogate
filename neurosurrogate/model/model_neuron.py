from collections import Counter
from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass
class Node:
    name: str
    type: str  # "hh" | "passive"


@dataclass
class Edge:
    src: str
    dst: str
    weight: float


@dataclass(frozen=False)
class NeuronGraph:
    nodes: list[Node]
    edges: list[Edge]
    stim: str  # node name

    @cached_property
    def _name_to_idx(self) -> dict:
        return {n.name: i for i, n in enumerate(self.nodes)}

    @property
    def names(self) -> list[str]:
        return [n.name for n in self.nodes]

    @property
    def types(self) -> list[str]:
        return [n.type for n in self.nodes]

    def name_to_idx(self, name: str) -> int:
        return self._name_to_idx[name]

    @property
    def connections(self):
        return [
            (self.name_to_idx(e.src), self.name_to_idx(e.dst), e.weight)
            for e in self.edges
        ]

    @property
    def stim_node_idx(self) -> int:
        return self.name_to_idx(self.stim)

    def to_dict(self) -> dict:
        return {
            "nodes": [{"name": n.name, "type": n.type} for n in self.nodes],
            "edges": [
                {"src": e.src, "dst": e.dst, "weight": e.weight} for e in self.edges
            ],
            "stim": self.stim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NeuronGraph":
        nodes = [Node(name=n["name"], type=n["type"]) for n in d["nodes"]]
        edges = [
            Edge(src=e["src"], dst=e["dst"], weight=e["weight"]) for e in d["edges"]
        ]
        return cls(nodes=nodes, edges=edges, stim=d["stim"])

    @property
    def graph_laplacian(self):
        connections = self.connections
        N = len(self.nodes)
        G_matrix = np.zeros((N, N), dtype=np.float64)
        if N == 1 or connections is None:
            pass
        else:
            for i, j, g in connections:
                G_matrix[i, j] = G_matrix[j, i] = g
        D_matrix = np.diag(np.sum(G_matrix, axis=1))
        C_matrix = G_matrix - D_matrix  # 流入を正とするグラフラプラシアンの符号反転

        return C_matrix


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
