"""サロゲートの主体。

`SurrogateBundle` が学習の同定情報 (meta) と成果物 (preprocessor / closure) を
保持し、定式化 (ansatz/) を差し替えながら学習・保存を駆動するオーケストレーター。
ansatz は状態を持たないストラテジで、**bundle 自身ではなく meta / preprocessor /
closure を受け取る** (オーケストレーターへ依存を張り返さない)。

学習 (`setup`: simulate → preprocessor build → 閉包項の同定) と `load` が別経路
なので、load は保存された 3 点を戻すだけで済む。学習データは保存せず meta から
lazy に再現する (`train_xr`) → load 後でも触れて、参照しなければ simulate は走らない。
"""

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import xarray as xr

from ..core.network import CompartmentType
from ..core.opcost import OpCost
from .ansatz.base import Ansatz
from .ansatz.hybrid import HybridAnsatz
from .ansatz.sindy import SINDyAnsatz
from .ansatz.ude import UDEAnsatz
from .closure.base import Closure
from .meta import SurrogateMeta
from .preprocessor.autoencoder import AEPreprocessor
from .preprocessor.base import Preprocessor
from .preprocessor.pca import PCAPreprocessor

BUNDLE_FILE = "surrogate.joblib"  # 学習成果物 (closure/preprocessor)
META_FILE = "meta.json"  # 同定情報。一覧はこれだけ読む

# meta の dispatch キー → 実装。**解決するのは bundle だけ**なので、実装側に type 名
# を持たせず (自分がどう選ばれたかを知らない) ここに対応表を置く。
SURR_CLS: dict[str, type[Ansatz[Any]]] = {
    "sindy": SINDyAnsatz,
    "hybrid": HybridAnsatz,
    "ude": UDEAnsatz,
}
PREPROCESSOR_CLS: dict[str, type[Preprocessor]] = {
    "pca": PCAPreprocessor,
    "ae": AEPreprocessor,
}


class SurrogateBundle:
    """サロゲート本体。meta / preprocessor / closure を持ち ansatz へ委譲する。

    属性は 3 つとも setup / load が代入して埋める (`__init__` 引数は取らない —
    埋まる時点が違うだけで meta も他と同格)。未設定のまま参照すれば AttributeError
    で早期に気付く。
    """

    meta: SurrogateMeta
    preprocessor: Preprocessor
    closure: Closure

    @cached_property
    def train_xr(self) -> xr.Dataset:
        """学習データ。実体は保存せず meta から決定的に再現する (dataset/電流/dt が
        meta に揃っている)。→ load 経路でも参照でき、marimo は run をロードするたび
        に `closure.train_source` と合わせて学習入力を組み直して描ける。"""
        return self.meta.simulate()

    @cached_property
    def ansatz(self) -> Ansatz[Any]:
        """定式化ストラテジ。meta.surrogate_type から解決する (状態なし → 保存不要)。"""
        return SURR_CLS[self.meta.surrogate_type]()

    @cached_property
    def preprocessor_cls(self) -> type[Preprocessor]:
        """preprocessor 実装。ansatz と同じく meta の dispatch キーから解決する
        (解決だけが cached_property、学習済みインスタンスは属性 `preprocessor`)。"""
        return PREPROCESSOR_CLS[self.meta.preprocessor_type]

    # --- 構築 ---------------------------------------------------------------

    @classmethod
    def setup(cls, cfg: dict) -> "SurrogateBundle":
        """設定ツリーから学習済み bundle を組む唯一の入口。

        cfg の 3 ブロックは各構成要素の構築引数そのもので、bundle は宛先へ振り分け
        学習順に走らせるだけ (設定を組み替えない = 構造への暗黙依存を持たない):
          meta         → `SurrogateMeta.build` (学習構造 = 実装の dispatch キー)
          preprocessor → `preprocessor_cls.fit` (種別固有 hyperparams のみ)
          ansatz       → `ansatz.fit`           (定式化固有 hyperparams のみ)
        """
        bundle = cls()
        bundle.meta = SurrogateMeta.build(**cfg["meta"])
        bundle.preprocessor = bundle.preprocessor_cls.fit(
            bundle.ansatz.train_source(bundle.meta).stacked_gate(bundle.train_xr),
            bundle.meta.n_components,
            cfg["preprocessor"],
        )
        bundle.closure = bundle.ansatz.fit(
            bundle.meta, bundle.train_xr, bundle.preprocessor, cfg["ansatz"]
        )
        return bundle

    @classmethod
    def load_meta(cls, dir: Path | str) -> SurrogateMeta:
        """同定情報だけを読む。run 一覧は pickle を開かずに済む。"""
        return SurrogateMeta.from_dict(json.loads((Path(dir) / META_FILE).read_text()))

    @classmethod
    def load(cls, dir: Path | str) -> "SurrogateBundle":
        data = joblib.load(Path(dir) / BUNDLE_FILE)
        bundle = cls()
        bundle.meta = cls.load_meta(dir)
        bundle.preprocessor = data["preprocessor"]
        bundle.closure = data["closure"]
        return bundle

    def save(self, dir: Path | str) -> None:
        """meta は JSON (構造で残す → クラス定義に縛られない)、学習成果物は pickle。"""
        (Path(dir) / META_FILE).write_text(
            json.dumps(self.meta.to_dict(), indent=2, ensure_ascii=False)
        )
        joblib.dump(
            {"closure": self.closure, "preprocessor": self.preprocessor},
            Path(dir) / BUNDLE_FILE,
        )

    # --- ansatz 委譲 --------------------------------------------------------

    @property
    def surr_comp_type(self) -> CompartmentType:
        """置換後の CompartmentType (replace.apply_surrogate が差し込む)。"""
        return self.ansatz.surr_comp_type(self.meta, self.preprocessor, self.closure)

    @property
    def opcost(self) -> OpCost:
        return self.ansatz.opcost(self.meta, self.preprocessor, self.closure)

    def metrics(self) -> dict:
        return {
            **self.closure.metrics(),
            **self.preprocessor.metrics(),
            **self.opcost.diff_dict(self.meta.original_opcost),
        }
