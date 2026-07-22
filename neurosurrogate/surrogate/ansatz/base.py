from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ...core.network import CompartmentType
from ...core.opcost import OpCost
from ..sindy import SINDyBundle

if TYPE_CHECKING:
    from ..bundle import SurrogateBundle


class Ansatz(ABC):
    """方程式の定式化: 列構造・kernel・演算コストをどう組むか。

    **状態を持たない**。学習設定も成果物も SurrogateBundle が持ち、ansatz は bundle
    を受け取って計算するだけのストラテジ (1 インスタンスを bundle が使い回す)。
    「方程式の形を知るのが ansatz、データと成果物を持つのが bundle」の分担。
    """

    SURROGATE_TYPE: ClassVar[str]

    @abstractmethod
    def train_gate(self, bundle: "SurrogateBundle") -> np.ndarray:
        """preprocessor 学習に使うゲート行列 (ansatz が列選択)。setup が参照する。"""
        ...

    @abstractmethod
    def fit(
        self, bundle: "SurrogateBundle", optimizer: dict, library_specs: list[dict]
    ) -> SINDyBundle:
        """SINDy 同定 (列構造は ansatz が Roles で決める)。"""
        ...

    @abstractmethod
    def surr_comp_type(self, bundle: "SurrogateBundle") -> CompartmentType:
        """置換後の surrogate CompartmentType (学習結果から構築)。"""
        ...

    @abstractmethod
    def opcost(self, bundle: "SurrogateBundle") -> OpCost:
        """置換後 kernel 1 ステップの演算コスト (構成が定式化ごとに違う)。"""
        ...
