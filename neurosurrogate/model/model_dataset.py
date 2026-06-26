from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import hydra
import numpy as np

from ..builder.registry_current import FUNC_MAP
from ..profiler.profiler_model import OpCost


class Compartment:
    def __init__(
        self,
        type_name: str,
        gate_inits: list[float],
        gate_names: list[str],
        name: str = "",
        v_init: float = -65,
        OpCost: "OpCost | None" = None,
    ):
        self.name = name
        self.type_name = type_name
        self.v_init = v_init
        self.gate_inits = gate_inits
        self.gate_names = gate_names
        self._opcost = OpCost

    def with_name(self, name: str) -> "Compartment":
        return Compartment(
            type_name=self.type_name,
            gate_inits=self.gate_inits,
            gate_names=self.gate_names,
            name=name,
            v_init=self.v_init,
            OpCost=self._opcost,
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type_name}

    @classmethod
    def from_dict(cls, d: dict) -> "Compartment":
        from .registry_compartments import COMPARTMENT_TEMPLATES

        return COMPARTMENT_TEMPLATES[d["type"]].with_name(d["name"])

    @property
    def vars(self):
        return ["V"] + self.gate_names

    @property
    def gate(self):
        return [False] + [True] * len(self.gate_names)

    @property
    def init(self):
        return [self.v_init] + self.gate_inits

    @property
    def OpCost(self):
        return self._opcost


@dataclass
class CurrentConfig:
    pipeline: dict

    def build(self, dt: float) -> np.ndarray:
        return hydra.utils.instantiate(self.pipeline)(dt)

    @staticmethod
    def build_pipeline(current_type: str, kw: dict) -> dict:
        return {
            "_target_": f"neurosurrogate.builder.registry_current.{FUNC_MAP[current_type].__name__}",
            **kw,
        }

    def to_dict(self) -> dict:
        return {"pipeline": self.pipeline}

    @classmethod
    def from_dict(cls, d: dict) -> "CurrentConfig":
        return cls(pipeline=d["pipeline"])


@dataclass
class Edge:
    src: str
    dst: str
    weight: float


@dataclass(frozen=False)
class NeuronGraph:
    nodes: list[Compartment]
    edges: list[Edge]
    stim: str  # node name

    @cached_property
    def _name_to_idx(self) -> dict:
        return {c.name: i for i, c in enumerate(self.nodes)}

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.nodes]

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
            "nodes": [c.to_dict() for c in self.nodes],
            "edges": [
                {"src": e.src, "dst": e.dst, "weight": e.weight} for e in self.edges
            ],
            "stim": self.stim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NeuronGraph":
        return cls(
            nodes=[Compartment.from_dict(n) for n in d["nodes"]],
            edges=[Edge(**e) for e in d["edges"]],
            stim=d["stim"],
        )

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
        return G_matrix - np.diag(
            np.sum(G_matrix, axis=1)
        )  # 流入を正とするグラフラプラシアンの符号反転

    def with_surrogates(
        self,
        targets: set[str],
        make_surr: Callable[[str], "Compartment"],
    ) -> "NeuronGraph":
        nodes = [make_surr(n.name) if n.name in targets else n for n in self.nodes]
        return NeuronGraph(nodes=nodes, edges=self.edges, stim=self.stim)

    @staticmethod
    def chain(
        node_types: list[str],
        weights: list[float],
        stim: int = 0,
    ) -> "NeuronGraph":
        """
        ノード名は型の頭文字 + 0始まり連番で自動生成
        例: ["passive", "hh", "passive"] → ["p0", "h0", "p1"]
        """
        from .registry_compartments import COMPARTMENT_TEMPLATES

        assert len(weights) == len(node_types) - 1, (
            f"weights の長さは len(node_types) - 1 = {len(node_types) - 1} である必要があります"
        )
        counters: Counter = Counter()
        nodes = []
        for t in node_types:
            prefix = t[0]  # "hh" → "h", "passive" → "p"
            nodes.append(
                COMPARTMENT_TEMPLATES[t].with_name(f"{prefix}{counters[prefix]}")
            )
            counters[prefix] += 1

        return NeuronGraph(
            nodes=nodes,
            edges=[
                Edge(nodes[i].name, nodes[i + 1].name, w) for i, w in enumerate(weights)
            ],
            stim=nodes[stim].name,
        )


@dataclass
class DatasetConfig:
    model_name: str
    dt: float
    current: CurrentConfig
    net: NeuronGraph

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dt": self.dt,
            "current": self.current.to_dict(),
            "net": self.net.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        return cls(
            model_name=d["model_name"],
            dt=d["dt"],
            current=CurrentConfig.from_dict(d["current"]),
            net=NeuronGraph.from_dict(d["net"]),
        )

    def with_surrogates(
        self,
        targets: set[str],
        make_surr: Callable[[str], "Compartment"],
    ) -> "DatasetConfig":
        return DatasetConfig(
            model_name=self.model_name,
            dt=self.dt,
            current=self.current,
            net=self.net.with_surrogates(targets, make_surr),
        )

    @classmethod
    def build_dataset(
        cls,
        dt: float,
        model_name: str,
        pipeline: dict,
    ) -> "DatasetConfig":
        """yamlとの境界"""
        from .registry_neuron import MCMODELS

        return DatasetConfig(
            model_name=model_name,
            dt=dt,
            current=CurrentConfig(pipeline=pipeline),
            net=MCMODELS[model_name],
        )
