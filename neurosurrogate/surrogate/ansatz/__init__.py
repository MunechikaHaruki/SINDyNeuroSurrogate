from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import sympy as sp
import xarray as xr

from ...core import access
from ...core.network import CompartmentType
from ..closure import Closure
from ..preprocessor import Preprocessor

if TYPE_CHECKING:
    from ..model import SurrogateSpec

C = TypeVar("C", bound=Closure)


@dataclass(frozen=True)
class TrainInputs:
    """同定器へ渡す直前の入力一式 (列順 = ansatz が組んだ列構造)。fit が作って流し、
    view が同じものを描く。軌道は comp ごとに分けたまま持つ (縦連結は偽微分)。時間軸と
    出所 comp は持たない (training_data / scope.train_comp_ids が源)。

    x_names/u_names : 列の表示名 (状態列 / 入力列)。
    """

    x_names: list[str]
    u_names: list[str]
    x: list[np.ndarray]  # 各 (time, len(x_names))、comp_ids 順
    u: list[np.ndarray]  # 各 (time, len(u_names))

    def target_symbols(self) -> list[sp.Symbol]:
        """状態列の記号 (列名がそのまま記号)。"""
        return [sp.Symbol(v) for v in self.x_names]

    def input_symbols(self) -> list[sp.Symbol]:
        """入力列の記号。"""
        return [sp.Symbol(v) for v in self.u_names]


class Ansatz(ABC, Generic[C]):
    """方程式の定式化 (列構造・kernel・演算コストの組み方)。

    状態を持たないストラテジ。設定も成果物も Surrogate が持ち、ansatz は
    spec / training_data / preprocessor / closure を引数で受けて計算するだけ
    (Surrogate 自身は受けない = オーケストレーターへ依存を張り返さない)。

    型引数 C = 同定する閉包項の具体型。ξ の行割り / NN 呼びなど型固有の引き出しは
    `Closure` 契約に載らないので、ここで具体型に束縛する。
    """

    @abstractmethod
    def n_train_gate(self, spec: SurrogateSpec) -> int:
        """先頭から学習するゲート本数 (残りは physics)。定式化ごとに違う唯一の学習範囲
        — comp 選択は定式化に依らず `scope.train_comp_ids` が共通で組む。"""
        ...

    def training_gates(
        self, spec: SurrogateSpec, training_data: xr.Dataset
    ) -> list[np.ndarray]:
        """学習 comp ごとの学習対象ゲート。軌道は分けたまま返す。"""
        return [
            gate[:, : self.n_train_gate(spec)]
            for gate in access.gate_matrices(training_data, spec.train_comp_ids())
        ]

    @abstractmethod
    def train_inputs(
        self,
        spec: SurrogateSpec,
        training_data: xr.Dataset,
        preprocessor: Preprocessor,
    ) -> TrainInputs:
        """同定器へ渡す直前の (x, u) を組む。fit が流し view が描く。"""
        ...

    @abstractmethod
    def fit(
        self,
        spec: SurrogateSpec,
        training_data: xr.Dataset,
        preprocessor: Preprocessor,
        config: dict,
    ) -> C:
        """閉包項を同定する。config = 定式化固有の hyperparams のみ (共通の潜在次元は
        spec 側)。"""
        ...

    @abstractmethod
    def surr_comp_type(
        self,
        spec: SurrogateSpec,
        preprocessor: Preprocessor,
        closure: C,
    ) -> CompartmentType:
        """置換後の CompartmentType (学習結果から構築)。演算コストは元コンパートメント
        と同じく `opcost` フィールドへ焼き込む (surr だけ別経路を持たせない)。"""
        ...
