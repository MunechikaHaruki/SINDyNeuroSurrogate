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
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import xarray as xr

from ..core.network import CompartmentType
from ..sim.spec import EvalSeries
from .ansatz.base import Ansatz
from .ansatz.impl.hybrid import HybridAnsatz
from .ansatz.impl.sindy import SINDyAnsatz
from .ansatz.impl.ude import UDEAnsatz
from .closure.base import Closure
from .meta import SurrogateMeta
from .preprocessor.base import Preprocessor
from .preprocessor.impl.autoencoder import AEPreprocessor
from .preprocessor.impl.pca import PCAPreprocessor
from .replace import applicable

_BUNDLE_FILE = "surrogate.joblib"  # 学習成果物 (closure/preprocessor)
META_FILE = "meta.json"  # 同定情報。一覧はこれだけ読む

# meta の dispatch キー → 実装。**解決するのは bundle だけ**なので、実装側に type 名
# を持たせず (自分がどう選ばれたかを知らない) ここに対応表を置く。
_SURR_CLS: dict[str, type[Ansatz[Any]]] = {
    "sindy": SINDyAnsatz,
    "hybrid": HybridAnsatz,
    "ude": UDEAnsatz,
}
_PREPROCESSOR_CLS: dict[str, type[Preprocessor]] = {
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
        return _SURR_CLS[self.meta.surrogate_type]()

    @cached_property
    def preprocessor_cls(self) -> type[Preprocessor]:
        """preprocessor 実装。ansatz と同じく meta の dispatch キーから解決する
        (解決だけが cached_property、学習済みインスタンスは属性 `preprocessor`)。"""
        return _PREPROCESSOR_CLS[self.meta.preprocessor_type]

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
    def load(cls, dir: Path | str) -> "SurrogateBundle":
        # meta は JSON 別ファイル (構造で保存)、学習成果物は pickle。run 一覧が meta
        # だけ読む経路は mlflow_io が artifact の meta.json を直読みする (bundle を
        # 経由しない) → ここは load 内でまとめて読めば足りる。
        data = joblib.load(Path(dir) / _BUNDLE_FILE)
        bundle = cls()
        bundle.meta = SurrogateMeta.from_dict(
            json.loads((Path(dir) / META_FILE).read_text())
        )
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
            Path(dir) / _BUNDLE_FILE,
        )

    # --- ansatz 委譲 --------------------------------------------------------

    @property
    def surr_comp_type(self) -> CompartmentType:
        """置換後の CompartmentType (replace.apply_surrogate が差し込む)。"""
        return self.ansatz.surr_comp_type(self.meta, self.preprocessor, self.closure)


@dataclass(frozen=True)
class SurrogateRuns:
    """一意なrun名と選択順を保ったsurrogate列。

    MLflowのrun IDはI/O層だけの関心。この型はrun名の一意性、pathの1区切りとしての
    有効性、結果のrun軸との対応を保証し、名前を凡例・行見出し・保存段にそのまま使う。
    """

    runs: tuple[tuple[str, SurrogateBundle], ...]

    def __post_init__(self) -> None:
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError(f"学習 run 名が重複 {names}")
        for name in names:
            if not name or name in (".", "..") or any(c in name for c in "/\\\0"):
                raise ValueError(f"学習 run 名をpathに使えない {name!r}")

    def __iter__(self) -> Iterator[tuple[str, SurrogateBundle]]:
        return iter(self.runs)

    def __len__(self) -> int:
        return len(self.runs)

    @property
    def names(self) -> tuple[str, ...]:
        """選択順の学習run名。"""
        return tuple(name for name, _ in self.runs)

    def replacing(self, series: EvalSeries) -> "SurrogateRuns":
        """この掃引を実際に置換できる run だけの列。**run 軸を絞る唯一の場所**。

        置換できない run は落ちる (回しても比較にならない)。空になった選択を飛ばすか
        拒むかは呼び出し側の関心 (その場で回す側は飛ばし、保存側は拒む)。
        """
        return SurrogateRuns(
            tuple(
                (name, bundle)
                for name, bundle in self
                if applicable(bundle.meta, series)
            )
        )

    def bundle(self, name: str) -> SurrogateBundle:
        """学習run名からsurrogateを引く。"""
        for candidate, bundle in self:
            if candidate == name:
                return bundle
        raise KeyError(f"run {name!r} がsurrogate列に無い")
