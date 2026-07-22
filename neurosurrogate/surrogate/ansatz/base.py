from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import xarray as xr

from ...core import access
from ...core.network import CompartmentType
from ...core.opcost import OpCost
from ..closure.base import Closure
from ..meta import SurrogateMeta
from ..preprocessor.base import Preprocessor
from ..replace import train_comp_ids

C = TypeVar("C", bound=Closure)


@dataclass(frozen=True)
class TrainSource:
    """学習入力の**選択規則** (どの comp の・先頭何本のゲートを食わせるか)。

    meta だけで決まるので保存しない — 学習成果物 (closure) に焼き付けなくても
    `Ansatz.train_source(meta)` でいつでも引き直せる。preprocessor 学習は encode
    前のゲートを要るので、encode 済みの `TrainInputs` では代替できない (両者は
    「規則」と「同定器へ渡す実体」の関係)。

    comp_ids : 軌道を取る comp (comp_id 昇順)。
    n_gate   : 先頭から何本のゲートを学習するか (残りは physics 側が解く)。
    """

    comp_ids: list[int]
    n_gate: int

    def gate(self, train_xr: xr.Dataset, comp_id: int) -> np.ndarray:
        """comp_id の学習ゲート行列 (time, n_gate)。"""
        return access.gate_matrix(train_xr, comp_id)[:, : self.n_gate]

    def stacked_gate(self, train_xr: xr.Dataset) -> np.ndarray:
        """全 comp を縦連結した学習ゲート行列 (preprocessor 学習用。preprocessor は
        時間微分を取らないので連結してよい — 閉包項の同定は軌道を分けて渡す)。"""
        return np.concatenate([self.gate(train_xr, i) for i in self.comp_ids], axis=0)


@dataclass(frozen=True)
class TrainInputs:
    """同定器へ渡す**直前**の入力一式 (列順 = ansatz が組んだ列構造そのもの)。

    fit がこれを作って同定器へ流し、view は同じものを描く → 図と学習入力が食い違わ
    ない (fit のローカルを view が組み直さない)。軌道は comp ごとに分けたまま持つ
    (縦連結すると境界に偽の時間微分が入る)。時間軸は train_xr と共通なので持たない。
    軌道の出所 comp も持たない — 選択規則は `TrainSource.comp_ids` が唯一の源で、
    ここは「その規則で組んだ実体」に徹する (同じ列を 2 か所に置かない)。

    x_names/u_names : 列の表示名 (状態列 / 入力列)。
    """

    x_names: list[str]
    u_names: list[str]
    x: list[np.ndarray]  # 各 (time, len(x_names))。TrainSource.comp_ids と同順
    u: list[np.ndarray]  # 各 (time, len(u_names))


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
    def n_train_gate(self, meta: SurrogateMeta) -> int:
        """先頭から何本のゲートを学習するか (残りは physics 側が解く)。

        定式化ごとに違う唯一の学習範囲。**どの comp から取るかは定式化に依らない**
        (置換対象全部 or 指定 1 ノード) ので `train_source` が共通で組む。
        """
        ...

    def train_source(self, meta: SurrogateMeta) -> TrainSource:
        """学習入力の列選択 (どの comp の・先頭何ゲートを食わせるか)。

        meta だけで決まる → setup (preprocessor 学習) も fit も view も、必要な
        ときに引き直せば同じものが得られる (成果物に保存しない)。
        """
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
        """同定器へ渡す直前の (x, u) を組む。fit がそのまま流し、view が描く。"""
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
