"""学習の同定情報: 何を・どのデータで・どのノードから学習したか。

bundle / ansatz / replace が共通で参照する **leaf** (surrogate 内の他モジュールに
依存しない)。ansatz と replace はここと preprocessor だけを見れば足り、
オーケストレーターである SurrogateBundle を知らずに済む。
"""

from dataclasses import dataclass

from ..core.network import Compartment, CompartmentType, DatasetConfig
from ..core.opcost import OpCost
from ..core.simulator import unified_simulator


@dataclass(frozen=True)
class SurrogateMeta:
    """何を学習したかの同定情報 (学習構造・学習データ・学習元ノード)。

    surrogate_type / preprocessor_type は実装を解決する dispatch キー、
    n_components は潜在次元。**学習構造の単一源**で、実装側 (ansatz/preprocessor)
    は自分がどう選ばれたかを知らない。
    """

    surrogate_type: str  # sindy/hybrid
    preprocessor_type: str  # pca/ae
    n_components: int
    dataset: DatasetConfig
    train_comp_id: int

    @classmethod
    def build(
        cls,
        surrogate_type: str,
        preprocessor_type: str,
        n_components: int,
        datasets: dict,
        train_comp_identifier: str,
    ) -> "SurrogateMeta":
        dataset = DatasetConfig.build_dataset(**datasets)
        return cls(
            surrogate_type=surrogate_type,
            preprocessor_type=preprocessor_type,
            n_components=n_components,
            dataset=dataset,
            train_comp_id=dataset.net.name_to_idx(train_comp_identifier),
        )

    @property
    def label(self) -> str:
        """図表示用の簡約名。例 hybrid/n2/ae。runName 文字列に非依存。"""
        return f"{self.surrogate_type}/n{self.n_components}/{self.preprocessor_type}"

    @property
    def train_comp(self) -> Compartment:
        return self.dataset.net.nodes[self.train_comp_id]

    @property
    def train_comp_type(self) -> CompartmentType:
        """学習元コンパートメントの物理型 (= 置換対象)。"""
        return self.train_comp.type

    @property
    def original_opcost(self) -> OpCost | None:
        return self.train_comp.type.opcost

    def simulate(self):
        return unified_simulator(self.dataset)
