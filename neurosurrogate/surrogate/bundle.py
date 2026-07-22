"""サロゲートの主体。

`SurrogateBundle` が学習の同定情報 (meta) と成果物 (preprocessor / sindy_bundle) を
保持し、定式化 (ansatz/) を差し替えながら学習・保存を駆動するオーケストレーター。
ansatz は状態を持たないストラテジで、**bundle 自身ではなく meta / preprocessor /
sindy_bundle を受け取る** (オーケストレーターへ依存を張り返さない)。

学習 (`setup`: simulate → preprocessor build → SINDy 同定) と `load` が別経路なので、
load は保存された 3 点を戻すだけで済み simulate は走らない。
"""

from functools import cached_property
from pathlib import Path

import joblib
import xarray as xr

from ..core.network import CompartmentType
from ..core.opcost import OpCost
from .ansatz.base import Ansatz
from .meta import SurrogateMeta
from .preprocessor import PREPROCESSOR_CLS, Preprocessor
from .sindy import SINDyBundle

BUNDLE_FILE = "surrogate.joblib"


class SurrogateBundle:
    """サロゲート本体。meta / preprocessor / sindy_bundle を持ち ansatz へ委譲する。

    属性は 4 つとも setup / load が代入して埋める (`__init__` 引数は取らない —
    埋まる時点が違うだけで meta も他と同格)。未設定のまま参照すれば AttributeError
    で早期に気付く。train_xr は学習にしか要らないので保存せず load 経路では未設定。
    """

    meta: SurrogateMeta
    preprocessor: Preprocessor
    sindy_bundle: SINDyBundle
    train_xr: xr.Dataset

    @cached_property
    def ansatz(self) -> Ansatz:
        """定式化ストラテジ。meta.surrogate_type から解決する (状態なし → 保存不要)。"""
        from .ansatz import SURR_CLS

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
          sindy        → `ansatz.fit`
        """
        bundle = cls()
        bundle.meta = SurrogateMeta.build(**cfg["meta"])
        bundle.train_xr = bundle.meta.simulate()
        bundle.preprocessor = bundle.preprocessor_cls.fit(
            bundle.ansatz.train_gate(bundle.meta, bundle.train_xr),
            bundle.meta.n_components,
            cfg["preprocessor"],
        )
        bundle.sindy_bundle = bundle.ansatz.fit(
            bundle.meta, bundle.train_xr, bundle.preprocessor, **cfg["sindy"]
        )
        return bundle

    @classmethod
    def load(cls, dir: Path | str) -> "SurrogateBundle":
        data = joblib.load(Path(dir) / BUNDLE_FILE)
        bundle = cls()
        bundle.meta = data["meta"]
        bundle.preprocessor = data["preprocessor"]
        bundle.sindy_bundle = data["sindy_bundle"]
        return bundle

    def save(self, dir: Path | str) -> None:
        joblib.dump(
            {
                "meta": self.meta,
                "sindy_bundle": self.sindy_bundle,
                "preprocessor": self.preprocessor,
            },
            Path(dir) / BUNDLE_FILE,
        )

    # --- ansatz 委譲 --------------------------------------------------------

    @property
    def surr_comp_type(self) -> CompartmentType:
        """置換後の CompartmentType (replace.apply_surrogate が差し込む)。"""
        return self.ansatz.surr_comp_type(
            self.meta, self.preprocessor, self.sindy_bundle
        )

    @property
    def opcost(self) -> OpCost:
        return self.ansatz.opcost(self.meta, self.preprocessor, self.sindy_bundle)

    def metrics(self) -> dict:
        return {
            **self.sindy_bundle.xi_metrics(),
            **self.preprocessor.metrics(),
            **self.opcost.diff_dict(self.meta.original_opcost),
        }
