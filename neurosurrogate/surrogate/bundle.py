"""サロゲートの主体。

`SurrogateBundle` が学習設定 (meta) と成果物 (preprocessor / sindy_bundle) を保持し、
定式化 (ansatz/) を差し替えながら学習・保存・置換を駆動するオーケストレーター。
ansatz は状態を持たないストラテジで、bundle を受け取って計算するだけ
(「方程式の形を知るのが ansatz、データと成果物を持つのが bundle」)。

学習セットアップ (`setup`: simulate → preprocessor build) と `load` が別経路なので、
load は保存された 3 点を戻すだけで済み simulate は走らない。
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import joblib
import xarray as xr

from ..core.network import (
    Compartment,
    CompartmentType,
    DatasetConfig,
)
from ..core.opcost import OpCost
from ..core.simulator import unified_simulator
from .ansatz.base import Ansatz
from .preprocessor import PREPROCESSOR_CLS, Preprocessor
from .sindy import SINDyBundle

BUNDLE_FILE = "surrogate.joblib"


@dataclass(frozen=True)
class SurrogateMeta:
    surrogate_type: str
    dataset: DatasetConfig
    train_comp_id: int
    n_components: int
    preprocessor_kind: str  # pca/ae

    @classmethod
    def build(
        cls,
        surrogate_type: str,
        datasets: dict,
        train_comp_identifier: str,
        n_components: int,
        preprocessor_kind: str,
    ) -> "SurrogateMeta":
        dataset = DatasetConfig.build_dataset(**datasets)
        return cls(
            surrogate_type=surrogate_type,
            dataset=dataset,
            train_comp_id=dataset.net.name_to_idx(train_comp_identifier),
            n_components=n_components,
            preprocessor_kind=preprocessor_kind,
        )

    @property
    def label(self) -> str:
        """図表示用の簡約名。例 hybrid/n2/ae。runName 文字列に非依存。"""
        return f"{self.surrogate_type}/n{self.n_components}/{self.preprocessor_kind}"

    @property
    def train_comp(self) -> Compartment:
        return self.dataset.net.nodes[self.train_comp_id]

    @property
    def train_comp_type(self) -> CompartmentType:
        """学習元コンパートメントの物理型 (= 置換対象)。"""
        return self.train_comp.type

    @property
    def original_opcost(self) -> OpCost | None:
        return self.train_comp.type.opcost

    def simulate(self):
        return unified_simulator(self.dataset)


class SurrogateBundle:
    """サロゲート本体。meta / preprocessor / sindy_bundle を持ち ansatz へ委譲する。

    preprocessor は setup 時、sindy_bundle は fit 時、train_xr は setup 時に埋まる
    (学習の進行順)。未設定のまま参照すれば AttributeError で早期に気付く。train_xr
    は fit にしか要らないので保存せず、load 経路では未設定のまま。
    """

    preprocessor: Preprocessor
    sindy_bundle: SINDyBundle
    train_xr: xr.Dataset

    def __init__(self, meta: SurrogateMeta) -> None:
        self.meta = meta

    @cached_property
    def ansatz(self) -> Ansatz:
        """定式化ストラテジ。meta.surrogate_type から解決する (状態なし → 保存不要)。"""
        from .ansatz import SURR_CLS

        return SURR_CLS[self.meta.surrogate_type]()

    # --- 構築 ---------------------------------------------------------------

    @classmethod
    def setup(
        cls,
        type: str,
        datasets: dict,
        train_comp_identifier: str,
        n_components: int,
        preprocessor: dict,
    ) -> "SurrogateBundle":
        """学習セットアップ: meta 構築 → simulate → ansatz が列選択したゲートで
        preprocessor build まで。`fit` は残りの SINDy 同定のみを担う (責務二層分離)。
        学習構造 (n_components / preprocessor 種別) は meta が単一源で保持し save で
        永続化する。
        """
        bundle = cls(
            SurrogateMeta.build(
                surrogate_type=type,
                datasets=datasets,
                train_comp_identifier=train_comp_identifier,
                n_components=n_components,
                preprocessor_kind=preprocessor["type"],
            )
        )
        bundle.train_xr = bundle.meta.simulate()
        # preprocessor spec: type が dispatch キー、残りが種別固有 hyperparams。
        # n_components は meta が単一源なのでここで注入する。
        spec = dict(preprocessor)
        bundle.preprocessor = PREPROCESSOR_CLS[spec.pop("type")].fit(
            bundle.ansatz.train_gate(bundle), {**spec, "n_components": n_components}
        )
        return bundle

    @classmethod
    def load(cls, dir: Path | str) -> "SurrogateBundle":
        data = joblib.load(Path(dir) / BUNDLE_FILE)
        bundle = cls(data["meta"])
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

    def fit(self, optimizer: dict, library_specs: list[dict]) -> None:
        self.sindy_bundle = self.ansatz.fit(self, optimizer, library_specs)

    @property
    def surr_comp_type(self) -> CompartmentType:
        """置換後の CompartmentType (replace.apply_surrogate が差し込む)。"""
        return self.ansatz.surr_comp_type(self)

    @property
    def opcost(self) -> OpCost:
        return self.ansatz.opcost(self)

    def metrics(self) -> dict:
        return {
            **self.sindy_bundle.xi_metrics(),
            **self.preprocessor.metrics(),
            **self.opcost.diff_dict(self.meta.original_opcost),
        }
