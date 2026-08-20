"""`Surrogate` が差し替える 3 構成要素の**契約をここに集約する**。

`Closure` (学習成果物) / `Preprocessor` (ゲート↔潜在変換) / `Ansatz` (方程式の骨格)
の 3 つは互いを参照し合う (Ansatz は両者を受けて kernel を組み、型引数で Closure に
束縛される) ので、契約は 1 モジュールに置く = パッケージ間の抽象レベルの依存辺を
持たない。**実装は各サブパッケージ** (`ansatz/` `closure/` `preprocessor/`) に並び、
`from .. import Closure` のように上から契約を引く。

3 つは対等ではない: `closure` / `preprocessor` が leaf、`ansatz` が両者を合成する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import jax.numpy as jnp
import numpy as np
import sympy as sp
import xarray as xr

from ...core import access
from ...core.network import CompartmentType
from ...core.opcost import OpCost

if TYPE_CHECKING:
    from ..model import SurrogateSpec


class Closure(ABC):
    @abstractmethod
    def metrics(self) -> dict[str, float]:
        """MLflow へ流すモデル指標 (表現ごとに中身は違う)。bundle が型を知らない
        まま呼ぶ窓口。"""
        ...

    @abstractmethod
    def opcost(self) -> OpCost:
        """閉包項 1 回の評価コスト。構成は表現ごとに違うが問い方は引数なしで同一 →
        全実装が持ち ansatz が一律に呼ぶので契約に載せる。"""
        ...


def _reconstruction_stats(
    encode: Callable, decode: Callable, train_gate: np.ndarray
) -> tuple[float, float]:
    reconstructed = np.asarray(decode(encode(train_gate)))
    mse = float(np.mean((train_gate - reconstructed) ** 2))
    return mse, mse / float(np.var(train_gate))


class Preprocessor(ABC):
    """ゲート ↔ 潜在の可逆変換。学習結果を np で保持し直列化可能。"""

    # 以下は _set_fit_artifacts が fit 末尾で設定する (__init__ 引数ではない)。
    reconstruction_mse: float
    reconstruction_mse_ratio: float
    # 学習データ先頭の潜在 = 置換シミュの初期ゲート値。
    gate_inits: list

    @classmethod
    @abstractmethod
    def fit(cls, train_gate: np.ndarray, n_components: int, spec: dict) -> Preprocessor:
        """潜在次元 n_components (全種共通) と spec (種別固有 hyperparams) で学習。"""
        ...

    @abstractmethod
    def encode(self, x: np.ndarray) -> np.ndarray:
        """ゲート → 潜在 (診断 / 学習データ変換)。"""
        ...

    @abstractmethod
    def decode(self, state: jnp.ndarray) -> jnp.ndarray:
        """潜在 → ゲート (kernel で毎ステップ呼ぶ)。"""
        ...

    @abstractmethod
    def metrics(self) -> dict: ...

    @abstractmethod
    def opcost(self) -> OpCost:
        """decode 1 回の演算コスト (hybrid kernel の decode 分)。"""
        ...

    @property
    @abstractmethod
    def n_features(self) -> int:
        """encode 入力のゲート数 (transform_gate の幅整合用)。"""
        ...

    def _set_fit_artifacts(self, train_gate: np.ndarray) -> None:
        """encode/decode 確定後に再構成統計と初期潜在を埋める (fit 末尾で呼ぶ)。"""
        self.reconstruction_mse, self.reconstruction_mse_ratio = _reconstruction_stats(
            self.encode, self.decode, train_gate
        )
        self.gate_inits = self.encode(train_gate)[0].tolist()


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
    def params_match(self, train: tuple | None, node: tuple | None) -> bool:
        """学習ノードの回路 params で同定したものを、params が `node` のノードへ
        適用してよいか。**定式化そのものの性質** (回路 params を入力として受ける
        形なら不変、係数へ焼き込む形なら一致が要る) なので契約に載せる。
        `SurrogateSpec.in_train_domain` が学習ドメインの判定に使う。"""
        ...

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
