from dataclasses import dataclass
from functools import cached_property

import hydra
import numpy as np

from ..builder.build_current import FUNC_MAP


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
            "_target_": f"neurosurrogate.builder.build_current.{FUNC_MAP[current_type].__name__}",
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
        from .model_neuron import MCMODELS

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
