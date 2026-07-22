"""学習の同定情報: 何を・どのデータで・どのノードから学習したか。

bundle / ansatz / replace が共通で参照する **leaf** (surrogate 内の他モジュールに
依存しない)。ansatz と replace はここと preprocessor だけを見れば足り、
オーケストレーターである SurrogateBundle を知らずに済む。
"""

from dataclasses import dataclass

import xarray as xr

from ..compartments import COMPARTMENT_TYPES
from ..core.network import Compartment, CompartmentType, DatasetConfig
from ..core.opcost import OpCost
from ..core.simulator import unified_simulator


@dataclass(frozen=True)
class SurrogateMeta:
    """何を学習したかの同定情報 (学習構造・置換対象の種類・学習データ)。

    surrogate_type / preprocessor_type は実装を解決する dispatch キー、
    n_components は潜在次元。**学習構造の単一源**で、実装側 (ansatz/preprocessor)
    は自分がどう選ばれたかを知らない。

    置換対象と学習範囲は別軸で、yaml もその 2 キーで指定する:
      `comp_type`     … **置換対象のコンパートメントの種類** (yaml 直指定)。
                        サロゲートは「種類 → それを置換するモデル」の対応であって
                        MC ネットワーク名やノード名には紐づかない → 置換判定
                        (replace の型一致) も実装 dispatch (hybrid の physics) も
                        ここだけを見る。
      `train_comp_id` … **学習軌道を 1 ノードへ絞る指定** (yaml の
                        `train_comp_identifier`、既定 None)。None なら種類一致の
                        置換対象ノード全部で学習する (訓練分布=評価分布)。
    """

    surrogate_type: str  # sindy/hybrid
    preprocessor_type: str  # pca/ae
    n_components: int
    dataset: DatasetConfig
    comp_type: CompartmentType  # 置換対象の種類 (hh/traub…)
    train_comp_id: int | None  # 学習を絞るノード。None = 種類一致ノード全部

    @classmethod
    def build(
        cls,
        surrogate_type: str,
        preprocessor_type: str,
        n_components: int,
        datasets: dict,
        comp_type: str,
        train_comp_identifier: str | None = None,
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
        )

    @property
    def label(self) -> str:
        """図表示用の簡約名。例 hybrid/n2/ae@traub19。runName 文字列に非依存。

        末尾は学習データの MC モデル名。学習構造が同じでも学習データが違えば別物
        (traub 単体で学習 vs traub19 の全 comp で学習) → sweep はこれを識別キーに
        するので、データ名まで含めないと別 run が silent に 1 本へ潰れる。
        """
        return f"{self.surrogate_type}/n{self.n_components}/{self.preprocessor_type}"

    @property
    def surr_type_name(self) -> str:
        """置換後 CompartmentType の名前。例 traub_hybrid_surr。

        `simulator._group_by_type` はノードを **type 名でバケット化し代表 1 個の
        kernel を全員へ適用する** → 名前が衝突すると片方の kernel が黙って使われる。
        由来 (種類 × 定式化) を名前へ入れておけば、置換後ノードが何由来か図やログ
        から読めるうえ、別サロゲート同士が同名になることもない。
        """
        return f"{self.comp_type.name}_{self.surrogate_type}_surr"

    @property
    def train_comp(self) -> Compartment:
        """学習の基準ノード (params/初期値の参照先)。

        絞り込み指定があればそのノード、無ければ dataset 内で種類が一致する先頭
        ノード。sindy は学習時 params をモデルへ焼き込むので、params 両立の比較対象
        (replace の `_PARAMS_MATCH`) と置換後の初期電位はここから取る。置換可否の
        「種類」判定はここを経由しない (comp_type が直接持つ)。
        """
        if self.train_comp_id is not None:
            return self.dataset.net.nodes[self.train_comp_id]
        return next(n for n in self.dataset.net.nodes if n.type == self.comp_type)

    @property
    def original_opcost(self) -> OpCost | None:
        """置換前 1 ステップのコスト = 置換対象の種類が持つコスト。"""
        return self.comp_type.opcost

    def simulate(self) -> xr.Dataset:
        return unified_simulator(self.dataset)
