from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import xarray as xr

from ...core.network import CompartmentType
from ...core.opcost import OpCost
from ..closure.base import Closure
from ..meta import SurrogateMeta
from ..preprocessor.base import Preprocessor

C = TypeVar("C", bound=Closure)


class Ansatz(ABC, Generic[C]):
    """方程式の定式化: 列構造・kernel・演算コストをどう組むか。

    **状態を持たない**。学習設定も成果物も SurrogateBundle が持ち、ansatz は必要な
    ものを引数で受け取って計算するだけのストラテジ (1 インスタンスを使い回す)。
    受けるのは meta / train_xr / preprocessor / closure であって bundle 自身では
    ない — オーケストレーターへ依存を張り返さないため。
    「方程式の形を知るのが ansatz、データと成果物を持つのが bundle」の分担。

    型引数 C = 自分が同定する閉包項の具体型。閉包項から何をどう引き出すか (ξ の
    行を割る / NN を呼ぶ / 評価コストを聞く) は定式化ごとに違い `Closure` の契約
    には載らない → ここで具体型に束縛して受け取る。
    """

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
        spec: dict,
    ) -> C:
        """閉包項を同定する (列構造は ansatz が決める)。

        spec = 定式化固有の学習 hyperparams のみ (SINDy: optimizer/library_specs、
        UDE: epochs/lr/…)。潜在次元など全定式化共通のものは meta が持つ —
        `Preprocessor.fit(gate, n_components, spec)` と同じ切り分け。
        """
        ...

    @abstractmethod
    def surr_comp_type(
        self,
        meta: SurrogateMeta,
        preprocessor: Preprocessor,
        closure: C,
    ) -> CompartmentType:
        """置換後の surrogate CompartmentType (学習結果から構築)。"""
        ...

    @abstractmethod
    def opcost(
        self,
        meta: SurrogateMeta,
        preprocessor: Preprocessor,
        closure: C,
    ) -> OpCost:
        """置換後 kernel 1 ステップの演算コスト (構成が定式化ごとに違う)。"""
        ...
