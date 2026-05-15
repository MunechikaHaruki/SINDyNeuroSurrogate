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
    iteration: int
    silence_steps: int
    pipeline: dict

    def build(self):
        dset_i_ext = np.zeros(self.iteration)

        if (
            self.iteration - self.silence_steps <= self.silence_steps
        ):  # active_end <=active_start
            raise ValueError(
                f"silence_steps={self.silence_steps} が大きすぎます（iteration={self.iteration}）"
            )
        active = dset_i_ext[self.silence_steps : self.iteration - self.silence_steps]
        func = hydra.utils.instantiate(self.pipeline)
        func(active)
        return dset_i_ext

    @staticmethod
    def build_pipeline(current_type: str, kw: dict) -> dict:
        return {
            "_target_": f"neurosurrogate.builder.registry_current.{FUNC_MAP[current_type].__name__}",
            **kw,
        }

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "silence_steps": self.silence_steps,
            "pipeline": self.pipeline,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CurrentConfig":
        return cls(
            iteration=d["iteration"],
            silence_steps=d["silence_steps"],
            pipeline=d["pipeline"],
        )


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
        D_matrix = np.diag(np.sum(G_matrix, axis=1))
        C_matrix = G_matrix - D_matrix  # 流入を正とするグラフラプラシアンの符号反転

        return C_matrix

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
            name = f"{prefix}{counters[prefix]}"
            nodes.append(COMPARTMENT_TEMPLATES[t].with_name(name))
            counters[prefix] += 1

        edges = [
            Edge(nodes[i].name, nodes[i + 1].name, w) for i, w in enumerate(weights)
        ]
        return NeuronGraph(nodes=nodes, edges=edges, stim=nodes[stim].name)


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

    @classmethod
    def build_dataset(
        cls,
        dt: float,
        silence_duration: float,
        duration: float,
        model_name: str,
        pipeline: dict,
    ) -> "DatasetConfig":
        """yamlとの境界"""
        from .registry_neuron import MCMODELS

        return DatasetConfig(
            model_name=model_name,
            dt=dt,
            current=CurrentConfig(
                iteration=int(duration / dt),
                silence_steps=int(silence_duration / dt),
                pipeline=pipeline,
            ),
            net=MCMODELS[model_name],
        )
