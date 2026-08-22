"""サロゲートの主体と学習仕様。

`Surrogate` が学習仕様 (spec) と成果物 (preprocessor / closure) を保持する。定式化
(ansatz) は spec しか状態を持たないので field でなく派生 — 定式化に依る問い
(学習ゲート・列構造・置換後の型) は `surrogate.ansatz` へ直接問う。
**この型は転送メソッドを持たない**。

**組み立ては `Surrogate` のメソッドでない** — 型が答えるのは持つ・保存する・読むの
3 つだけで、設定ツリーから組む手順はモジュール関数 `fit_surrogate` に置く。
設定の形を解くのは `SurrogateSpec.from_config` 1 箇所。
**置換 (どこへ当てられるか・当てる) は `replace.py`** — 学習済みのものを持つことと、
それを別のネットへ当てることは別の概念で、後者だけが適用先 (`NeuronGraph`) を知る。
`in_train_domain` / `train_comp_ids` がここに残るのは、それが学習ドメインという
**spec 自身の性質**で、置換の可否と学習 comp の選定の両方から引かれるから。
学習データは保存せず spec から lazy に再現する (`training_data`)。load 後でも触れ、
参照しなければ simulate は走らない。
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import xarray as xr

from ..core.network import Compartment, CompartmentType
from ..core.opcost import OpCost
from ..core.simulator import unified_simulator
from ..neurons import COMPARTMENT_TYPES
from ..sim.spec import SimSpec
from .parts import Ansatz, Closure, Preprocessor
from .parts.ansatz.hybrid import HybridSINDyAnsatz
from .parts.ansatz.sindy import SINDyAnsatz
from .parts.ansatz.ude import UDEAnsatz
from .parts.preprocessor.autoencoder import fit_ae
from .parts.preprocessor.pca import fit_pca

_BUNDLE_FILE = "surrogate.joblib"  # 学習成果物 (closure/preprocessor)
SPEC_FILE = "spec.json"  # 同定情報。一覧はこれだけ読む

# spec の dispatch キー → 実装。**解決するのは Surrogate だけ**なので、実装側に type 名
# を持たせず (自分がどう選ばれたかを知らない) ここに対応表を置く。
_SURR_CLS: dict[str, type[Ansatz[Any]]] = {
    "sindy": SINDyAnsatz,
    "hybrid": HybridSINDyAnsatz,
    "ude": UDEAnsatz,
}
# 前処理は**型でなく学習関数**を引く: 種別ごとに hyperparams が違うので、名前で解いた
# 先に config をそのまま展開して渡せる形 (`Callable[..., Preprocessor]`) にする。
# 引数が静的に検査されないのは ansatz と同じ = 実行時の文字列で選ぶ以上そこに知識が
# 無いというだけで、受理するキーと既定値は各関数の署名が持つ。
_PREPROCESSOR_FIT: dict[str, Callable[..., Preprocessor]] = {
    "pca": fit_pca,
    "ae": fit_ae,
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
    # `parts/` の 3 構成要素と 1 対 1 の hyperparams。**共通のつまみ (n_components)
    # だけは上の field へ昇格**していて、ここには**その層の入口 1 つの署名へそのまま
    # 展開されるもの**だけが残る (preprocessor→`fit_pca`/`fit_ae`、
    # ansatz→`HybridAnsatz.split`、closure→閉包項の同定入口)。層をまたいで持たない =
    # ある層の実装を替えても他の層の値は動かない。
    # 学習中しか読まれないが、仕様は学習後も由来として残るものなので spec が持つ。
    preprocessor_config: dict
    ansatz_config: dict
    closure_config: dict

    # --- 構築・直列化 -------------------------------------------------------
    # 素通しでないのは dataset (入れ子の構造) と comp_type (名前 ↔ 実体) の 2 つだけ
    # なので、その 2 つだけ明示して残りは field 走査に任せる = 仕様に項目を足すとき
    # 触るのは上の宣言 1 箇所。

    @classmethod
    def from_config(cls, config: dict) -> "SurrogateSpec":
        """Hydra の surrogate ブロックから。**4 ブロックはそのまま仕様の一部**で、
        ここが設定ツリーの形を知る唯一の場所 (以降 fit は spec しか見ない)。

        学習ノードは config では名前で書き、ここで index へ解く。
        """
        spec = config["spec"]
        dataset = SimSpec(**spec["datasets"])
        train_comp_identifier = spec.get("train_comp_identifier")
        return cls(
            surrogate_type=spec["surrogate_type"],
            preprocessor_type=spec["preprocessor_type"],
            n_components=spec["n_components"],
            dataset=dataset,
            comp_type=COMPARTMENT_TYPES[spec["comp_type"]],
            train_comp_id=(
                None
                if train_comp_identifier is None
                else dataset.net.name_to_idx(train_comp_identifier)
            ),
            preprocessor_config=config["preprocessor"],
            ansatz_config=config["ansatz"],
            closure_config=config["closure"],
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

    def ansatz(self) -> type[Ansatz[Any]]:
        """定式化の実装。**dispatch キーを解くのはここだけ**で、返すのはインスタンス
        でなく型 — `Ansatz` は状態を持たず全メソッドが spec を引数で受けるので、
        学習前 (`in_train_domain` の `params_match`) も学習後も入口はこれ 1 つ。"""
        return _SURR_CLS[self.surrogate_type]

    def preprocessor_fit(self) -> Callable[..., Preprocessor]:
        """座標変換の学習関数。ansatz と同じく dispatch キーはここでだけ解く
        (実際に呼ぶのは `fit_surrogate`)。"""
        return _PREPROCESSOR_FIT[self.preprocessor_type]

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
        — 対象は適用側が名前で明示し (`EvalSeries.replace_targets` / `replace.replace`
        の targets)、この述語はその 1 つ 1 つが通るかを答えるだけ。学習 comp の選定
        (`train_comp_ids`) も同じ述語を引く = 置換専用でないのでここに置く。
        """
        if comp.type != self.comp_type:
            return False
        return self.ansatz().params_match(
            self.train_comp().resolved_params, comp.resolved_params
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
    """**学習済み**サロゲート。spec / preprocessor / closure の 3 点を持つ。

    どう学習されたかはここの関心ではない (組むのは `fit_surrogate`)、どこへ当てるかも
    ここの関心ではない (置換するのは `replace.replace`) — この型は「持つ・保存する・
    読む」だけを担い、**設定ツリーも適用先も一切知らない**。
    持つ 3 点は**保存する 3 点そのもの** (spec.json + pickle 2 点)。

    `eq=False` = 同一性で比べる。中身は numpy を抱えるので値比較は意味を持たない。
    """

    spec: SurrogateSpec
    preprocessor: Preprocessor
    closure: Closure

    @property
    def ansatz(self) -> type[Ansatz[Any]]:
        """定式化 (型そのもの。`Ansatz` は状態を持たない)。定式化に依る問い
        (学習ゲート・列構造・置換後の型) は**この型を素通しせず**
        `surrogate.ansatz.f(surrogate.spec, ...)` と直接問う。"""
        return self.spec.ansatz()

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
        return cls(
            SurrogateSpec.read(Path(dir) / SPEC_FILE),
            data["preprocessor"],
            data["closure"],
        )

    def save(self, dir: Path | str) -> None:
        """spec は JSON (構造で残す → クラス定義に縛られない)、学習成果物は pickle。"""
        (Path(dir) / SPEC_FILE).write_text(
            json.dumps(self.spec.to_dict(), indent=2, ensure_ascii=False)
        )
        joblib.dump(
            {"closure": self.closure, "preprocessor": self.preprocessor},
            Path(dir) / _BUNDLE_FILE,
        )


def fit_surrogate(cfg: dict) -> Surrogate:
    """設定ツリーから学習済み surrogate を組む唯一の入口。

    **メソッドでなくモジュール関数**なのは、組み立てが `Surrogate` の関心でないから
    (この型が答えるのは持つ・保存する・読む・適用するの 4 つだけ)。設定の形を解くのは
    `SurrogateSpec.from_config` で、ここは simulate して学習させて組む手順だけを持つ。

    **座標変換 → 閉包項の順は定式化に依らない** (閉包項は潜在座標の上に立つ) ので、
    ansatz の既定実装でなくここで確定させる。ansatz へ問うのは定式化ごとに違う 2 点
    だけ — 何を学習ゲートとするか (`training_gates`) と、その座標で何を同定するか
    (`fit_closure`)。hyperparams は全部 spec が持つ = **spec に無い設定は効かない**。
    """
    spec = SurrogateSpec.from_config(cfg)
    ansatz = spec.ansatz()
    # 学習データは spec から決定的に再現できる (`Surrogate.training_data` と同じ式)
    # ので、学習済みモデルへ持ち回さず捨てる。
    training_data = unified_simulator(spec.dataset.materialize())
    preprocessor = spec.preprocessor_fit()(
        ansatz.training_gates(spec, training_data),
        spec.n_components,
        **spec.preprocessor_config,
    )
    closure = ansatz.fit_closure(spec, training_data, preprocessor)
    return Surrogate(spec, preprocessor, closure)
