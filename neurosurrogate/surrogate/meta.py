"""学習の同定情報: 何を・どのデータで・どのノードから学習したか。

bundle / ansatz / replace が共通で参照する **leaf** (surrogate 内の他モジュールに
依存しない)。ansatz と replace はここと preprocessor だけを見れば足り、
オーケストレーターである SurrogateBundle を知らずに済む。
"""

from dataclasses import dataclass, fields

import xarray as xr

from ..compartments import COMPARTMENT_TYPES
from ..core.network import Compartment, CompartmentType, DatasetConfig
from ..core.opcost import OpCost
from ..core.simulator import unified_simulator


@dataclass(frozen=True)
class SurrogateMeta:
    """何を学習したかの同定情報 (学習構造・置換対象の種類・学習データ)。

    surrogate_type / preprocessor_type は実装を解決する dispatch キー、n_components は
    潜在次元 = **学習構造の単一源** (実装側は自分がどう選ばれたかを知らない)。

    置換対象と学習範囲は別軸で yaml もその軸で指定する:
      comp_type     … 置換対象のコンパートメント種類。サロゲートは「種類 → それを置換
                      するモデル」で MC 名やノード名に紐づかない → 置換判定も hybrid の
                      physics dispatch もここだけ見る。
      train_comp_id … 学習を 1 ノードへ絞る指定 (None=種類一致ノード全部で学習)。
      physics_type  … 学習/physics の分割位置の変種 (None=comp_type 名)。
    """

    surrogate_type: str  # sindy/hybrid
    preprocessor_type: str  # pca/ae
    n_components: int
    dataset: DatasetConfig
    comp_type: CompartmentType  # 置換対象の種類 (hh/traub…)
    train_comp_id: int | None  # 学習を絞るノード。None = 種類一致ノード全部
    physics_type: str | None  # 学習/physics 分割の変種。None = comp_type 名

    @classmethod
    def build(
        cls,
        surrogate_type: str,
        preprocessor_type: str,
        n_components: int,
        datasets: dict,
        comp_type: str,
        train_comp_identifier: str | None = None,
        physics_type: str | None = None,
    ) -> "SurrogateMeta":
        dataset = DatasetConfig.build_dataset(**datasets)
        return cls(
            surrogate_type=surrogate_type,
            preprocessor_type=preprocessor_type,
            n_components=n_components,
            dataset=dataset,
            comp_type=COMPARTMENT_TYPES[comp_type],
            train_comp_id=(
                None
                if train_comp_identifier is None
                else dataset.net.name_to_idx(train_comp_identifier)
            ),
            physics_type=physics_type,
        )

    def to_dict(self) -> dict:
        """JSON 保存形。素データへ落とすのは comp_type/dataset だけ、残りはそのまま。"""
        return {
            **{f.name: getattr(self, f.name) for f in fields(self)},
            "comp_type": self.comp_type.name,
            "dataset": self.dataset.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SurrogateMeta":
        return cls(
            **{
                **d,
                "comp_type": COMPARTMENT_TYPES[d["comp_type"]],
                "dataset": DatasetConfig.from_dict(d["dataset"]),
            }
        )

    @property
    def label(self) -> str:
        """図凡例用の簡約名。条件の軸ごとに改行 (1 行だと凡例で潰れる)。例:

            hybrid/n5/ae
            +traub_sr_physics
            @traub19:soma

        `@` 以降 = 学習データ (MC モデル名 + 絞り込みノード)。学習構造が同じでも学習
        データが違えば別物 → sweep の識別キー。ここまで含めないと別 run が silent に
        1 本へ潰れる。既定値の軸は出さない。
        """
        return "\n".join(
            [
                f"{self.surrogate_type}/n{self.n_components}/{self.preprocessor_type}",
                *([] if self.physics_type is None else [f"+{self.physics_type}"]),
                "@"
                + self.dataset.model_name
                + ("" if self.train_comp_id is None else f":{self.train_comp.name}"),
            ]
        )

    @property
    def surr_type_name(self) -> str:
        """置換後 CompartmentType の名前 (例 traub_hybrid_surr)。

        simulator は type 名でノードをバケット化し代表 kernel を全員へ適用する → 名前
        衝突は片方の kernel を黙って捨てる。種類×定式化を名前へ入れて衝突回避 + 由来を
        図/ログから読めるようにする。
        """
        return f"{self.comp_type.name}_{self.surrogate_type}_surr"

    @property
    def train_comp(self) -> Compartment:
        """学習の基準ノード (params/初期値の参照先)。絞り込みがあればそのノード、
        無ければ種類一致の先頭。sindy は params を焼き込むので params 両立比較 (replace
        の `_PARAMS_MATCH`) と初期電位はここから取る。種類判定は経由しない。
        """
        nodes = self.dataset.net.nodes
        if self.train_comp_id is not None:
            return nodes[self.train_comp_id]  # 絞り込み指定あり → そのノード
        # 絞り込み無し → 種類一致ノードの先頭 (同種は params が全一致 → どれでも同じ)
        return next(n for n in nodes if n.type == self.comp_type)

    @property
    def original_opcost(self) -> OpCost | None:
        """置換前 1 ステップのコスト = 置換対象の種類が持つコスト。"""
        return self.comp_type.opcost

    def simulate(self) -> xr.Dataset:
        return unified_simulator(self.dataset)
