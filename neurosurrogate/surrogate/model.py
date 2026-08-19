"""サロゲートの主体。

`Surrogate` が学習仕様 (spec) と成果物 (preprocessor / closure) を
保持し、定式化 (ansatz/) を差し替えながら学習・保存を駆動するオーケストレーター。
ansatz は状態を持たないストラテジで、**Surrogate 自身ではなく spec / preprocessor /
closure を受け取る** (オーケストレーターへ依存を張り返さない)。

学習 (`fit`: simulate → preprocessor build → 閉包項の同定) と `load` が別経路
なので、load は保存された 3 点を戻すだけで済む。学習データは保存せず spec から
lazy に再現する (`training_data`)。load 後でも触れ、参照しなければ simulate は走らない。
"""

import json
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xarray as xr

from ..core.network import Compartment, CompartmentType, DatasetConfig, NeuronGraph
from ..core.opcost import OpCost
from ..core.simulator import unified_simulator
from ..neurons.compartments import COMPARTMENT_TYPES
from ..sim.spec import EvalSeries, SimSpec
from .parts import Ansatz, Closure, Preprocessor, TrainInputs
from .parts.ansatz.hybrid import HybridSINDyAnsatz
from .parts.ansatz.sindy import SINDyAnsatz
from .parts.ansatz.ude import UDEAnsatz
from .parts.preprocessor.autoencoder import AEPreprocessor
from .parts.preprocessor.pca import PCAPreprocessor

_BUNDLE_FILE = "surrogate.joblib"  # 学習成果物 (closure/preprocessor)
SPEC_FILE = "spec.json"  # 同定情報。一覧はこれだけ読む

# spec の dispatch キー → 実装。**解決するのは Surrogate だけ**なので、実装側に type 名
# を持たせず (自分がどう選ばれたかを知らない) ここに対応表を置く。
_SURR_CLS: dict[str, type[Ansatz[Any]]] = {
    "sindy": SINDyAnsatz,
    "hybrid": HybridSINDyAnsatz,
    "ude": UDEAnsatz,
}
_PREPROCESSOR_CLS: dict[str, type[Preprocessor]] = {
    "pca": PCAPreprocessor,
    "ae": AEPreprocessor,
}


@dataclass(frozen=True)
class SurrogateSpec:
    """config を正規化した学習仕様。仕様から一意な派生値もここで答える。"""

    surrogate_type: str
    preprocessor_type: str
    n_components: int
    dataset: SimSpec
    comp_type: CompartmentType
    train_comp_id: int | None
    physics_type: str | None

    def ansatz(self) -> Ansatz[Any]:
        """定式化ストラテジ。**dispatch キーを解くのはここだけ** (状態なしなので
        毎回作ってよい)。適用範囲の判定 (`replaceable`) も学習も同じ実体に問う。"""
        return _SURR_CLS[self.surrogate_type]()

    def surr_type_name(self) -> str:
        """置換後 CompartmentType の衝突しない名前。"""
        return f"{self.comp_type.name}_{self.surrogate_type}_surr"

    def train_comp(self) -> Compartment:
        """params・初期値を参照する学習基準ノード。"""
        if self.train_comp_id is not None:
            return self.dataset.net.nodes[self.train_comp_id]
        return next(
            node for node in self.dataset.net.nodes if node.type == self.comp_type
        )

    def original_opcost(self) -> OpCost | None:
        """置換前 CompartmentType の 1 ステップのコスト。"""
        return self.comp_type.opcost

    def replaceable(self, comp: Compartment) -> bool:
        """comp が適用範囲にあるか (種類一致かつ params 両立)。"""
        if comp.type != self.comp_type:
            return False
        return self.ansatz().params_match(
            self.train_comp().resolved_params, comp.resolved_params
        )

    def applicable(self, series: EvalSeries) -> bool:
        """系列の適用先に置換可能なノードが 1 つでもあるか。"""
        return any(self.replaceable(node) for node in series.spec.net.nodes)

    def train_comp_ids(self) -> list[int]:
        """学習軌道を取る comp id 列。既定は適用範囲全部、明示時だけ 1 ノード。"""
        if self.train_comp_id is not None:
            return [self.train_comp_id]
        return [
            i for i, comp in enumerate(self.dataset.net.nodes) if self.replaceable(comp)
        ]

    def replacement_targets(self, net: NeuronGraph) -> set[str]:
        """net 内の置換対象ノード名 (不整合・対象ゼロは fail first)。"""
        targets = {node.name for node in net.nodes if self.replaceable(node)}
        mismatched = [
            node
            for node in net.nodes
            if node.type == self.comp_type and node.name not in targets
        ]
        if mismatched:
            train = self.train_comp()
            raise ValueError(
                f"種類 {self.comp_type.name!r} 一致だが params 非両立のノード "
                f"{[node.name for node in mismatched]}: 学習ドメイン外。\n"
                f"  train({train.name}): {train.resolved_params}\n"
                + "\n".join(
                    f"  node({node.name}): {node.resolved_params}"
                    for node in mismatched
                )
            )
        if not targets:
            raise ValueError(
                f"種類 {self.comp_type.name!r} のノードが {net.names} に存在しない "
                "→ 置換対象ゼロ。適用不可"
            )
        return targets


def _build_spec(config: dict) -> SurrogateSpec:
    dataset = SimSpec(**config["datasets"])
    train_comp_identifier = config.get("train_comp_identifier")
    return SurrogateSpec(
        surrogate_type=config["surrogate_type"],
        preprocessor_type=config["preprocessor_type"],
        n_components=config["n_components"],
        dataset=dataset,
        comp_type=COMPARTMENT_TYPES[config["comp_type"]],
        train_comp_id=(
            None
            if train_comp_identifier is None
            else dataset.net.name_to_idx(train_comp_identifier)
        ),
        physics_type=config.get("physics_type"),
    )


def _spec_to_dict(spec: SurrogateSpec) -> dict:
    return {
        "surrogate_type": spec.surrogate_type,
        "preprocessor_type": spec.preprocessor_type,
        "n_components": spec.n_components,
        "dataset": spec.dataset.to_dict(),
        "comp_type": spec.comp_type.name,
        "train_comp_id": spec.train_comp_id,
        "physics_type": spec.physics_type,
    }


def _spec_from_dict(data: dict) -> SurrogateSpec:
    return SurrogateSpec(
        surrogate_type=data["surrogate_type"],
        preprocessor_type=data["preprocessor_type"],
        n_components=data["n_components"],
        dataset=SimSpec.from_dict(data["dataset"]),
        comp_type=COMPARTMENT_TYPES[data["comp_type"]],
        train_comp_id=data["train_comp_id"],
        physics_type=data["physics_type"],
    )


def read_spec(path: Path | str) -> SurrogateSpec:
    """保存済み JSON から学習仕様だけを読む。"""
    return _spec_from_dict(json.loads(Path(path).read_text()))


class Surrogate:
    """サロゲート本体。spec / preprocessor / closure を持ち ansatz へ委譲する。

    属性は 3 つとも fit / load が代入して埋める (`__init__` 引数は取らない —
    埋まる時点が違うだけで spec も他と同格)。未設定のまま参照すれば AttributeError
    で早期に気付く。
    """

    spec: SurrogateSpec
    preprocessor: Preprocessor
    closure: Closure

    @cached_property
    def training_data(self) -> xr.Dataset:
        """学習データ。実体は保存せず spec から決定的に再現する (dataset/電流/dt が
        spec に揃っている)。→ load 経路でも参照でき、marimo は run をロードするたび
        に学習範囲の規則を合わせて学習入力を組み直して描ける。"""
        return unified_simulator(self.spec.dataset.materialize())

    @cached_property
    def _ansatz(self) -> Ansatz[Any]:
        """定式化ストラテジ (spec が解決する。状態なし → 保存不要)。"""
        return self.spec.ansatz()

    @cached_property
    def _preprocessor_cls(self) -> type[Preprocessor]:
        """preprocessor 実装。ansatz と同じく spec の dispatch キーから解決する
        (解決だけが cached_property、学習済みインスタンスは属性 `preprocessor`)。"""
        return _PREPROCESSOR_CLS[self.spec.preprocessor_type]

    # --- 構築 ---------------------------------------------------------------

    @classmethod
    def fit(cls, cfg: dict) -> "Surrogate":
        """設定ツリーから学習済み surrogate を組む唯一の入口。

        cfg の 3 ブロックは各構成要素の構築引数そのもので、surrogate は宛先へ振り分け
        学習順に走らせるだけ (設定を組み替えない = 構造への暗黙依存を持たない):
          spec         → `_build_spec` (学習構造 = 実装の dispatch キー)
          preprocessor → `preprocessor_cls.fit` (種別固有 hyperparams のみ)
          ansatz       → `ansatz.fit`           (定式化固有 hyperparams のみ)
        """
        surrogate = cls()
        surrogate.spec = _build_spec(cfg["spec"])
        surrogate.preprocessor = surrogate._preprocessor_cls.fit(
            np.concatenate(
                surrogate._ansatz.training_gates(
                    surrogate.spec, surrogate.training_data
                ),
                axis=0,
            ),
            surrogate.spec.n_components,
            cfg["preprocessor"],
        )
        surrogate.closure = surrogate._ansatz.fit(
            surrogate.spec,
            surrogate.training_data,
            surrogate.preprocessor,
            cfg["ansatz"],
        )
        return surrogate

    @classmethod
    def load(cls, dir: Path | str) -> "Surrogate":
        # spec は JSON 別ファイル (構造で保存)、学習成果物は pickle。run 一覧が spec
        # だけ読む経路は mlflow_io が artifact の spec.json を直読みする (surrogate を
        # 経由しない) → ここは load 内でまとめて読めば足りる。
        data = joblib.load(Path(dir) / _BUNDLE_FILE)
        surrogate = cls()
        surrogate.spec = read_spec(Path(dir) / SPEC_FILE)
        surrogate.preprocessor = data["preprocessor"]
        surrogate.closure = data["closure"]
        return surrogate

    def save(self, dir: Path | str) -> None:
        """spec は JSON (構造で残す → クラス定義に縛られない)、学習成果物は pickle。"""
        (Path(dir) / SPEC_FILE).write_text(
            json.dumps(_spec_to_dict(self.spec), indent=2, ensure_ascii=False)
        )
        joblib.dump(
            {"closure": self.closure, "preprocessor": self.preprocessor},
            Path(dir) / _BUNDLE_FILE,
        )

    # --- ansatz 委譲 --------------------------------------------------------

    @property
    def n_training_gates(self) -> int:
        """先頭から学習するゲート本数。"""
        return self._ansatz.n_train_gate(self.spec)

    def training_gates(self) -> list[np.ndarray]:
        """学習 comp ごとの学習対象ゲート。"""
        return self._ansatz.training_gates(self.spec, self.training_data)

    def training_inputs(self) -> TrainInputs:
        """同定器へ渡した列名と軌道。"""
        return self._ansatz.train_inputs(
            self.spec, self.training_data, self.preprocessor
        )

    @property
    def surr_comp_type(self) -> CompartmentType:
        """置換後の CompartmentType。"""
        return self._ansatz.surr_comp_type(self.spec, self.preprocessor, self.closure)

    def apply(self, dataset: DatasetConfig) -> DatasetConfig:
        """学習ドメインに属す全ノードを surrogate に置換する。"""
        targets = self.spec.replacement_targets(dataset.net)
        return dc_replace(
            dataset,
            net=dc_replace(
                dataset.net,
                nodes=[
                    dc_replace(node, type=self.surr_comp_type)
                    if node.name in targets
                    else node
                    for node in dataset.net.nodes
                ],
            ),
        )
