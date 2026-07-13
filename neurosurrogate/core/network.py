from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np

from .opcost import OpCost


@dataclass(frozen=True)
class CompartmentType:
    """「hh とは何か」を集約した物理的な型定義。

    kernel + params class + gate 構造 + opcost を持つ。
    Compartment (グラフノードのインスタンス) はこの CompartmentType への参照を持つだけ。
    """

    name: str  # "hh", "passive", "traub", "surr"
    kernel: Callable  # (params, u_t, v, state) -> (dv, dstate)
    param_cls: "type | None"  # HHParams / PassiveParams / TraubParams / None (surr)
    default_params: "tuple | None"  # デフォルト NamedTuple / None
    gate_names: list[str]
    default_gate_inits: list[float]
    v_init: float = -65
    opcost: "OpCost | None" = None

    # --- 変数構造 (V + gates 組立)、type だけで決まる ---
    @property
    def vars(self) -> list[str]:
        return ["V"] + self.gate_names

    @property
    def gate(self) -> list[bool]:
        return [False] + [True] * len(self.gate_names)

    @property
    def init(self) -> list[float]:
        return [self.v_init] + self.default_gate_inits


@dataclass(frozen=True)
class Compartment:
    """
    グラフ内の1ノード。物理型 (CompartmentType) への参照と、カスタム params だけを持つ。
    """

    name: str
    type: CompartmentType
    params: "tuple | None" = None

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "type": self.type.name}
        if self.params is not None:
            d["params"] = self.params._asdict()  # type: ignore[attr-defined]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Compartment":
        from ..compartments import COMPARTMENT_TYPES

        comp_type = COMPARTMENT_TYPES[d["type"]]
        if "params" in d:
            assert comp_type.param_cls is not None
            params = comp_type.param_cls(**d["params"])
        else:
            params = None
        return cls(name=d["name"], type=comp_type, params=params)


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
    # 外部電流 u_ext を stim ノードに注入する前に乗じるスケール。
    # 密度 [μA/cm^2] スケールの u_ext を絶対 [μA] に変換する用途 (traub19 等)。
    # default 1.0 → 既存モデル (単位規約: 密度) 不変。
    stim_area_scale: float = 1.0

    @cached_property
    def _name_to_idx(self) -> dict[str, int]:
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
            "stim_area_scale": self.stim_area_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NeuronGraph":
        return cls(
            nodes=[Compartment.from_dict(n) for n in d["nodes"]],
            edges=[Edge(**e) for e in d["edges"]],
            stim=d["stim"],
            stim_area_scale=d.get("stim_area_scale", 1.0),
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

    def with_surrogate(
        self,
        new_type: CompartmentType,
        accept: Callable[["Compartment"], bool],
    ) -> "NeuronGraph":
        """accept が真のノードを new_type に置換 (妥当性ベース全置換)。

        置換可否の判定 (学習ドメイン照合) は呼び出し側 (surrogate) の責務。
        各ノードの name/params は保持し type だけ差し替える。
        """
        nodes = [replace(n, type=new_type) if accept(n) else n for n in self.nodes]
        return NeuronGraph(nodes=nodes, edges=self.edges, stim=self.stim)


@dataclass
class DatasetConfig:
    model_name: str
    dt: float
    current: dict  # {"type": str, "params": dict}
    net: NeuronGraph

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dt": self.dt,
            "current": self.current,
            "net": self.net.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        return cls(
            model_name=d["model_name"],
            dt=d["dt"],
            current=d["current"],
            net=NeuronGraph.from_dict(d["net"]),
        )

    def build_current(self) -> np.ndarray:
        from ..currents import CURRENT_MAP

        return CURRENT_MAP[self.current["type"]](**self.current.get("params", {}))(
            self.dt
        )

    def with_surrogate(
        self,
        new_type: CompartmentType,
        accept: Callable[["Compartment"], bool],
    ) -> "DatasetConfig":
        return DatasetConfig(
            model_name=self.model_name,
            dt=self.dt,
            current=self.current,
            net=self.net.with_surrogate(new_type, accept),
        )

    @classmethod
    def build_dataset(
        cls,
        dt: float,
        model_name: str,
        current: dict,
    ) -> "DatasetConfig":
        """yamlとの境界"""
        from ..models import MCMODELS

        return DatasetConfig(
            model_name=model_name,
            dt=dt,
            current=current,
            net=MCMODELS[model_name],
        )
