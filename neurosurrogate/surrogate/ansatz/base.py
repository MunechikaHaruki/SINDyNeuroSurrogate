from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import sympy as sp
import xarray as xr

from ...core import access
from ...core.network import CompartmentType
from ..closure.base import Closure
from ..meta import SurrogateMeta
from ..preprocessor.base import Preprocessor
from ..replace import train_comp_ids

C = TypeVar("C", bound=Closure)


@dataclass(frozen=True)
class TrainSource:
    """学習入力の選択規則。meta だけで決まり保存しない (`Ansatz.train_source` でいつでも
    引き直せる)。preprocessor 学習は encode 前のゲートを要るので `TrainInputs` では
    代替できない。

    comp_ids : 軌道を取る comp (昇順)。
    n_gate   : 先頭から学習するゲート本数 (残りは physics)。
    """

    comp_ids: list[int]
    n_gate: int

    def gate(self, train_xr: xr.Dataset, comp_id: int) -> np.ndarray:
        """comp_id の学習ゲート行列 (time, n_gate)。"""
        return access.gate_matrix(train_xr, comp_id)[:, : self.n_gate]

    def gates(self, train_xr: xr.Dataset) -> list[np.ndarray]:
        """選択 comp ごとの学習ゲート (comp_ids 順、縦連結は偽微分)。"""
        return [self.gate(train_xr, i) for i in self.comp_ids]

    def potentials(self, train_xr: xr.Dataset) -> list[np.ndarray]:
        """選択 comp ごとの電位 (time,)。物理 dV/dt を持つ定式化の入力列 V。"""
        return [access.potential(train_xr, i) for i in self.comp_ids]

    def stacked_gate(self, train_xr: xr.Dataset) -> np.ndarray:
        """全 comp を縦連結 (preprocessor 学習用。時間微分を取らないので連結可)。"""
        return np.concatenate(self.gates(train_xr), axis=0)


@dataclass(frozen=True)
class TrainInputs:
    """同定器へ渡す直前の入力一式 (列順 = ansatz が組んだ列構造)。fit が作って流し、
    view が同じものを描く。軌道は comp ごとに分けたまま持つ (縦連結は偽微分)。時間軸と
    出所 comp は持たない (train_xr / TrainSource.comp_ids が源)。

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

    状態を持たないストラテジ。設定も成果物も SurrogateBundle が持ち、ansatz は
    meta / train_xr / preprocessor / closure を引数で受けて計算するだけ (bundle 自身は
    受けない = オーケストレーターへ依存を張り返さない)。

    型引数 C = 同定する閉包項の具体型。ξ の行割り / NN 呼びなど型固有の引き出しは
    `Closure` 契約に載らないので、ここで具体型に束縛する。
    """

    @abstractmethod
    def n_train_gate(self, meta: SurrogateMeta) -> int:
        """先頭から学習するゲート本数 (残りは physics)。定式化ごとに違う唯一の学習範囲
        — comp 選択は定式化に依らず `train_source` が共通で組む。"""
        ...

    def train_source(self, meta: SurrogateMeta) -> TrainSource:
        """学習入力の選択 (meta だけで決まる → setup/fit/view が引き直せば同じ)。"""
        return TrainSource(
            comp_ids=train_comp_ids(meta), n_gate=self.n_train_gate(meta)
        )

    @abstractmethod
    def train_inputs(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
    ) -> TrainInputs:
        """同定器へ渡す直前の (x, u) を組む。fit が流し view が描く。"""
        ...

    @abstractmethod
    def fit(
        self,
        meta: SurrogateMeta,
        train_xr: xr.Dataset,
        preprocessor: Preprocessor,
        spec: dict,
    ) -> C:
        """閉包項を同定する。spec = 定式化固有の hyperparams のみ (共通の潜在次元は
        meta 側)。"""
        ...

    @abstractmethod
    def surr_comp_type(
        self,
        meta: SurrogateMeta,
        preprocessor: Preprocessor,
        closure: C,
    ) -> CompartmentType:
        """置換後の CompartmentType (学習結果から構築)。演算コストは元コンパートメント
        と同じく `opcost` フィールドへ焼き込む (surr だけ別経路を持たせない)。"""
        ...
