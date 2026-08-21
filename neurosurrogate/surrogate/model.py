"""サロゲートの主体と学習仕様。

`Surrogate` が学習仕様 (spec) と、それに束縛した定式化 (ansatz) と、成果物
(preprocessor / closure) を保持する。ansatz は spec を属性に持つので、定式化に依る
問い (学習ゲート・列構造・置換後の型) は `surrogate.ansatz` へ直接問う —
**この型は転送メソッドを持たない**。

**組み方はここに無い** (`fit.py`)。ここが持つのは学習済みのものを「保存する・読む・
適用する」だけで、設定ツリーの形を知らない = 設定の変更がこのモジュールへ届かない。
学習データは保存せず spec から lazy に再現する (`training_data`)。load 後でも触れ、
参照しなければ simulate は走らない。
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from dataclasses import replace as dc_replace
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import xarray as xr

from ..core.network import Compartment, CompartmentType, DatasetConfig, NeuronGraph
from ..core.opcost import OpCost
from ..core.simulator import unified_simulator
from ..neurons import COMPARTMENT_TYPES
from ..sim.spec import EvalSeries, SimSpec
from .parts import Ansatz, Closure, Preprocessor
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

    # --- 構築・直列化 -------------------------------------------------------
    # 素通しでないのは dataset (入れ子の構造) と comp_type (名前 ↔ 実体) の 2 つだけ
    # なので、その 2 つだけ明示して残りは field 走査に任せる = 仕様に項目を足すとき
    # 触るのは上の宣言 1 箇所。

    @classmethod
    def from_config(cls, config: dict) -> "SurrogateSpec":
        """Hydra の spec ブロックから。**学習ノードは config では名前で書き**、
        ここで index へ解く (以降 spec は index だけを持つ)。"""
        dataset = SimSpec(**config["datasets"])
        train_comp_identifier = config.get("train_comp_identifier")
        return cls(
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

    def to_dict(self) -> dict:
        """保存用の JSON 構造 (クラス定義に縛られない形で残す)。"""
        return {
            **{f.name: getattr(self, f.name) for f in fields(self)},
            "dataset": self.dataset.to_dict(),
            "comp_type": self.comp_type.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SurrogateSpec":
        """`to_dict` の逆。項目の過不足はそのまま TypeError (黙って既定で埋めない)。"""
        return cls(
            **{
                **data,
                "dataset": SimSpec.from_dict(data["dataset"]),
                "comp_type": COMPARTMENT_TYPES[data["comp_type"]],
            }
        )

    @classmethod
    def read(cls, path: Path | str) -> "SurrogateSpec":
        """保存済み JSON から学習仕様だけを読む (run 一覧はこれだけ読む)。"""
        return cls.from_dict(json.loads(Path(path).read_text()))

    # --- 実装の解決 ---------------------------------------------------------

    def ansatz_cls(self) -> type[Ansatz[Any]]:
        """定式化の実装クラス。**dispatch キーを解くのはここだけ**。学習前でも解けるので
        `in_train_domain` は束縛前のこれに `params_match` を問う。"""
        return _SURR_CLS[self.surrogate_type]

    def ansatz(self) -> Ansatz[Any]:
        """この仕様に束縛した定式化ストラテジ。"""
        return self.ansatz_cls()(self)

    def preprocessor_cls(self) -> type[Preprocessor]:
        """座標変換の実装クラス。ansatz と同じく dispatch キーはここでだけ解く
        (実際に呼ぶのは `Ansatz.fit`)。"""
        return _PREPROCESSOR_CLS[self.preprocessor_type]

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

    def in_train_domain(self, comp: Compartment) -> bool:
        """comp が学習ドメインに属すか (種類一致かつ params 両立)。

        **置換してよいかの判定はこれ 1 つ**。何を置換するかの選定はここでは決まらない
        — 対象は適用側が名前で明示し (`EvalSeries.replace_targets` /
        `Surrogate.apply` の targets)、この述語はその 1 つ 1 つが通るかを答えるだけ。
        """
        if comp.type != self.comp_type:
            return False
        return self.ansatz_cls().params_match(
            self.train_comp().resolved_params, comp.resolved_params
        )

    def rejected_targets(self, net: NeuronGraph, targets: Sequence[str]) -> list[str]:
        """明示指定のうち net 上で置換できないものの理由文 (空なら全部置換可)。

        通るかどうかは `in_train_domain` が答え、ここはそれが偽だった理由 (不在 /
        種類違い / params 非両立) を名前ごとに文にするだけ = **どの名前がなぜ通らな
        かったかを必ず名指しできる** (「何個か落ちた」で終わらせない)。

        理由が人の目に出るのは `Surrogate.apply` の ValueError 経由。run 選択
        (`SurrogateRuns.replacing`) は真偽だけを見て絞るので、そこで落ちた run の
        理由は表示されない — 表示したくなったらこの返り値を UI まで運ぶ。
        """
        reasons = []
        for name in targets:
            if name not in net.names:
                reasons.append(f"{name!r}: 適用先 {net.names} に存在しない")
                continue
            comp = net.nodes[net.name_to_idx(name)]
            if self.in_train_domain(comp):
                continue
            if comp.type != self.comp_type:
                reasons.append(
                    f"{name!r}: 種類 {comp.type.name!r} ≠ 学習した種類 "
                    f"{self.comp_type.name!r}"
                )
            else:
                train = self.train_comp()
                reasons.append(
                    f"{name!r}: params 非両立 = 学習ドメイン外\n"
                    f"    train({train.name}): {train.resolved_params}\n"
                    f"    node({name}): {comp.resolved_params}"
                )
        return reasons

    def applicable(self, series: EvalSeries) -> bool:
        """系列が置換対象に挙げたノードを**全部**置換できるか。

        部分一致は不可 = 「指定したうち通ったものだけ静かに置換」を作らない。
        """
        return bool(series.replace_targets) and not self.rejected_targets(
            series.spec.net, series.replace_targets
        )

    def train_comp_ids(self) -> list[int]:
        """学習軌道を取る comp id 列。既定は学習ネット内の学習ドメイン全部、明示時
        だけ 1 ノード。"""
        if self.train_comp_id is not None:
            return [self.train_comp_id]
        return [
            i
            for i, comp in enumerate(self.dataset.net.nodes)
            if self.in_train_domain(comp)
        ]


@dataclass(eq=False)
class Surrogate:
    """**学習済み**サロゲート。spec / ansatz / preprocessor / closure の 4 点を持つ。

    どう学習されたかはここの関心ではない (組むのは `fit.py`) — この型は
    「持つ・保存する・読む・適用する」だけを担い、**設定ツリーを一切知らない**。
    `ansatz` は spec に束縛済みなので、定式化に依る問い (学習ゲート・列構造・置換後の
    型) は**この型を素通しせず ansatz へ直接問う**。

    `eq=False` = 同一性で比べる。中身は numpy を抱えるので値比較は意味を持たない。
    """

    spec: SurrogateSpec
    ansatz: Ansatz[Any]
    preprocessor: Preprocessor
    closure: Closure

    @cached_property
    def training_data(self) -> xr.Dataset:
        """学習データ。実体は保存せず spec から決定的に再現する (dataset/電流/dt が
        spec に揃っている)。→ load 経路でも参照でき、marimo は run をロードするたび
        に学習範囲の規則を合わせて学習入力を組み直して描ける。"""
        return unified_simulator(self.spec.dataset.materialize())

    # --- 保存形式 (save/load は 1 つの契約の両半分) --------------------------

    @classmethod
    def load(cls, dir: Path | str) -> "Surrogate":
        # spec は JSON 別ファイル (構造で保存)、学習成果物は pickle。ansatz は状態を
        # 持たず spec から解けるので保存しない。run 一覧が spec だけ読む経路は
        # mlflow_io が artifact の spec.json を直読みする (surrogate を経由しない)
        # → ここは load 内でまとめて読めば足りる。
        data = joblib.load(Path(dir) / _BUNDLE_FILE)
        spec = SurrogateSpec.read(Path(dir) / SPEC_FILE)
        return cls(spec, spec.ansatz(), data["preprocessor"], data["closure"])

    def save(self, dir: Path | str) -> None:
        """spec は JSON (構造で残す → クラス定義に縛られない)、学習成果物は pickle。"""
        (Path(dir) / SPEC_FILE).write_text(
            json.dumps(self.spec.to_dict(), indent=2, ensure_ascii=False)
        )
        joblib.dump(
            {"closure": self.closure, "preprocessor": self.preprocessor},
            Path(dir) / _BUNDLE_FILE,
        )

    def apply(self, dataset: DatasetConfig, targets: Sequence[str]) -> DatasetConfig:
        """**明示指定された** targets を surrogate へ置換する。

        「互換なノードを全部」置換しない = 適用先の形態が変わっても置換範囲は動かない。
        1 つでも置換できなければ何も置換せず ValueError (部分適用を作らない)。
        """
        if not targets:
            raise ValueError("置換対象が空: 置換するノード名を明示指定すること")
        rejected = self.spec.rejected_targets(dataset.net, targets)
        if rejected:
            raise ValueError(
                f"{self.spec.surr_type_name()} で置換できない対象:\n  "
                + "\n  ".join(rejected)
            )
        names = set(targets)
        surr_comp_type = self.ansatz.surr_comp_type(self.preprocessor, self.closure)
        return dc_replace(
            dataset,
            net=dc_replace(
                dataset.net,
                nodes=[
                    dc_replace(node, type=surr_comp_type)
                    if node.name in names
                    else node
                    for node in dataset.net.nodes
                ],
            ),
        )
