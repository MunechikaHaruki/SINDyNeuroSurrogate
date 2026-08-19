from collections.abc import Callable
from dataclasses import dataclass
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
    gate_names: list[str]
    # params -> [V, *gates] 初期状態。初期値は params からの導出量 (Traub の Ca 濃度
    # XI は phi_area/g_Ca 依存、静止電位は V_LEAK/E_REST) なので型に定数で焼かず、
    # ノード自身の params で毎回解く。module 関数で束縛 (spec が pickle される)。
    inits: Callable[..., list[float]]
    opcost: "OpCost | None" = None

    # --- 変数構造 (V + gates 組立)、type だけで決まる ---
    @property
    def vars(self) -> list[str]:
        return ["V"] + self.gate_names

    @property
    def gate(self) -> list[bool]:
        return [False] + [True] * len(self.gate_names)


@dataclass(frozen=True)
class Compartment:
    """
    グラフ内の1ノード。物理型 (CompartmentType) への参照と、カスタム params だけを持つ。
    """

    name: str
    type: CompartmentType
    params: "tuple | None" = None

    @property
    def resolved_params(self) -> "tuple | None":
        """実効 params: 明示 params、無ければ型 default (param_cls())。

        置換の params 一致判定と初期状態の解決に使う共通基準。
        surr のように param_cls=None の型は params を持たず None。
        """
        if self.params is not None:
            return self.params
        return self.type.param_cls() if self.type.param_cls is not None else None

    @property
    def init(self) -> list[float]:
        """このノードの初期状態 [V, *gates]。type × 自身の params から毎回解く。

        導出量なのでフィールドに保存しない (replace の type 差替で stale 化する)。
        """
        return self.type.inits(self.resolved_params)


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


@dataclass
class DatasetConfig:
    """**実体化済みのシミュレーション入力**: 解いたネットと確定した電流波形。
    `unified_simulator` はこれだけを受け取る。

    仕様 (適用先の名前・電流の種類とパラメータ) は持たない — それは `spec.SimSpec`
    で、`SimSpec.materialize()` がここへ落とす。**名前 → 実体の解決を core に
    持ち込まない**ための分割で、おかげでこの層は他のディレクトリを一切 import
    しない (置換は `Surrogate.apply` が net を差し替えた複製を作る)。
    """

    dt: float
    net: NeuronGraph
    current: np.ndarray
