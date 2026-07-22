from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import xarray as xr

from ...core.network import CompartmentType
from ...core.opcost import OpCost
from ..meta import SurrogateMeta
from ..preprocessor import Preprocessor
from ..sindy import SINDyBundle


class Ansatz(ABC):
    """方程式の定式化: 列構造・kernel・演算コストをどう組むか。

    **状態を持たない**。学習設定も成果物も SurrogateBundle が持ち、ansatz は必要な
    ものを引数で受け取って計算するだけのストラテジ (1 インスタンスを使い回す)。
    受けるのは meta / train_xr / preprocessor / sindy_bundle であって bundle 自身
    ではない — オーケストレーターへ依存を張り返さないため。
    「方程式の形を知るのが ansatz、データと成果物を持つのが bundle」の分担。
    """

    SURROGATE_TYPE: ClassVar[str]

    @abstractmethod
    def train_gate(self, meta: SurrogateMeta, train_xr: xr.Dataset) -> np.ndarray:
        """preprocessor 学習に使うゲート行列 (ansatz が列選択)。setup が参照する。"""
        ...

    @abstractmethod
    def fit(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
        optimizer: dict,
        library_specs: list[dict],
    ) -> SINDyBundle:
        """SINDy 同定 (列構造は ansatz が Roles で決める)。"""
        ...

    @abstractmethod
    def surr_comp_type(
        self,
        meta: SurrogateMeta,
        preprocessor: Preprocessor,
        sindy_bundle: SINDyBundle,
    ) -> CompartmentType:
        """置換後の surrogate CompartmentType (学習結果から構築)。"""
        ...

    @abstractmethod
    def opcost(
        self,
        meta: SurrogateMeta,
        preprocessor: Preprocessor,
        sindy_bundle: SINDyBundle,
    ) -> OpCost:
        """置換後 kernel 1 ステップの演算コスト (構成が定式化ごとに違う)。"""
        ...
