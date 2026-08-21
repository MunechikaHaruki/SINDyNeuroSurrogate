"""`Surrogate` が差し替える 3 構成要素の**契約をここに集約する**。

`Closure` (学習成果物) / `Preprocessor` (ゲート↔潜在変換) / `Ansatz` (方程式の骨格)
の 3 つは互いを参照し合う (Ansatz は両者を受けて kernel を組み、型引数で Closure に
束縛される) ので、契約は 1 モジュールに置く = パッケージ間の抽象レベルの依存辺を
持たない。**実装は各サブパッケージ** (`ansatz/` `closure/` `preprocessor/`) に並び、
`from .. import Closure` のように上から契約を引く。

3 つは対等ではない: `closure` / `preprocessor` が leaf、`ansatz` が両者を合成する。

**ここに置くのは契約だけ** — 手続きは 1 行も持たない。全実装で同じ手続きになるもの
(fit 済み派生量の埋め方 = `preprocessor/fit_artifacts.py`、preprocessor→closure の
学習順 = `model.fit_surrogate`) も、既定実装としてここへ降ろさず持ち主の側へ置く。
受け渡しの型 `TrainInputs` も契約でないので `train_inputs.py` に分ける。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import jax.numpy as jnp
import numpy as np
import xarray as xr

from ...core.network import CompartmentType
from ...core.opcost import OpCost

if TYPE_CHECKING:
    from ..model import SurrogateSpec
    from .train_inputs import TrainInputs


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


class Preprocessor(ABC):
    """ゲート ↔ 潜在の可逆変換。学習結果を np で保持し直列化可能。"""

    # 学習データ先頭の潜在 = 置換シミュの初期ゲート値。__init__ 引数ではなく fit 末尾
    # で埋まる (encode 確定後にしか出せない)。**外から読まれる唯一の学習結果**なので
    # 契約に載せる (kernel の初期状態を組む ansatz が引く)。再構成統計の方は各実装の
    # `metrics()` の中でしか読まれない = 契約でなく実装の持ち物。
    gate_inits: list

    # **学習の入口は契約に載せない** (`fit` classmethod を持たない)。種別ごとに
    # hyperparams が違うので署名が揃わず、無理に揃えると dict を受けて実装で解く形に
    # なる = 受理するキーと既定値が署名から消える。代わりに各実装がモジュール関数
    # (`fit_pca` / `fit_ae`) を持ち、名前 → その関数の対応表は `model._PREPROCESSOR_FIT`
    # が持つ。契約が言うのは**学習済みの前処理に何ができるか**だけ。

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


C = TypeVar("C", bound=Closure)


class Ansatz(ABC, Generic[C]):
    """方程式の定式化 (列構造・kernel・演算コストの組み方)。

    **インスタンスを作らない** — 定式化は「どう方程式を組むか」という規則であって
    状態ではないので、全メソッドが `cls` と引数だけで答える純粋関数。学習仕様 (spec)
    も含めて必要なものは毎回引数で受ける = 束縛した spec と渡された spec が食い違う
    余地が構造的に無く、`SurrogateSpec.in_train_domain` のように**学習前・spec だけ
    の状態から問う**経路も特別扱いにならない (入口は `spec.ansatz()` 1 つ)。

    型引数 C = 同定する閉包項の具体型。ξ の行割り / NN 呼びなど型固有の引き出しは
    `Closure` 契約に載らないので、ここで具体型に束縛する。
    """

    @classmethod
    @abstractmethod
    def params_match(cls, train: tuple | None, node: tuple | None) -> bool:
        """学習ノードの回路 params で同定したものを、params が `node` のノードへ
        適用してよいか。**定式化そのものの性質** (回路 params を入力として受ける
        形なら不変、係数へ焼き込む形なら一致が要る) なので契約に載せ、spec にも
        学習結果にも依らない (spec すら受けない唯一のメソッド)。"""
        ...

    @classmethod
    @abstractmethod
    def n_train_gate(cls, spec: SurrogateSpec) -> int:
        """先頭から学習するゲート本数 (残りは physics)。定式化ごとに違う唯一の学習範囲
        — comp 選択は定式化に依らず `spec.train_comp_ids` が共通で組む。"""
        ...

    @classmethod
    @abstractmethod
    def training_gates(
        cls, spec: SurrogateSpec, training_data: xr.Dataset
    ) -> list[np.ndarray]:
        """学習 comp ごとの学習対象ゲート (軌道は分けたまま返す)。preprocessor を
        学習させる素データで、切り出し方は定式化の性質。"""
        ...

    @classmethod
    @abstractmethod
    def train_inputs(
        cls,
        spec: SurrogateSpec,
        training_data: xr.Dataset,
        preprocessor: Preprocessor,
    ) -> TrainInputs:
        """同定器へ渡す直前の (x, u) を組む。fit が流し view が描く。"""
        ...

    @classmethod
    @abstractmethod
    def fit_closure(
        cls,
        spec: SurrogateSpec,
        training_data: xr.Dataset,
        preprocessor: Preprocessor,
    ) -> C:
        """閉包項を同定する。preprocessor を書き換える実装もある (UDE の joint 学習)。

        hyperparams は spec の**その層のブロック**から取る — 閉包項の同定入口へ渡す
        なら `spec.closure_config`、定式化自身が回す学習なら `spec.ansatz_config`。
        """
        ...

    @classmethod
    @abstractmethod
    def surr_comp_type(
        cls, spec: SurrogateSpec, preprocessor: Preprocessor, closure: C
    ) -> CompartmentType:
        """置換後の CompartmentType (学習結果から構築)。演算コストは元コンパートメント
        と同じく `opcost` フィールドへ焼き込む (surr だけ別経路を持たせない)。"""
        ...
