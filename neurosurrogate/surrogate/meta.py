"""学習の同定情報: 何を・どのデータで・どのノードから学習したか。

bundle / ansatz / replace が共通で参照する **leaf** (surrogate 内の他モジュールに
依存しない)。ansatz と replace はここと preprocessor だけを見れば足り、
オーケストレーターである SurrogateBundle を知らずに済む。
"""

from dataclasses import dataclass

import xarray as xr

from ..core.network import Compartment, CompartmentType, DatasetConfig
from ..core.opcost import OpCost
from ..core.simulator import unified_simulator


@dataclass(frozen=True)
class SurrogateMeta:
    """何を学習したかの同定情報 (学習構造・置換対象の種類・学習データ)。

    surrogate_type / preprocessor_type は実装を解決する dispatch キー、
    n_components は潜在次元。**学習構造の単一源**で、実装側 (ansatz/preprocessor)
    は自分がどう選ばれたかを知らない。

    置換対象を指す 2 フィールドは役割が別で、重複していない:
      `comp_type`     … **コンパートメントの種類**。サロゲートは「種類 → それを
                        置換するモデル」の対応であって MC ネットワーク名やノード名
                        には紐づかない → 置換判定 (replace の型一致) と実装 dispatch
                        (hybrid の physics) はここだけを見る。yaml が渡すノード名は
                        build の入口で種類へ解決して捨てる。
      `train_comp_id` … **学習軌道を取った代表ノード**。sindy は学習時 params を
                        モデルへ焼き込む定式化なので、params 両立の比較対象
                        (replace の `_PARAMS_MATCH`) と置換後の初期電位の出所として
                        実ノードが要る。種類の同一性はここを経由しない。
    """

    surrogate_type: str  # sindy/hybrid
    preprocessor_type: str  # pca/ae
    n_components: int
    dataset: DatasetConfig
    comp_type: CompartmentType  # 置換対象の種類 (hh/traub…)
    train_comp_id: int  # 学習軌道の代表ノード (comp_type と一致)

    @classmethod
    def build(
        cls,
        surrogate_type: str,
        preprocessor_type: str,
        n_components: int,
        datasets: dict,
        train_comp_identifier: str,
    ) -> "SurrogateMeta":
        dataset = DatasetConfig.build_dataset(**datasets)
        train_comp_id = dataset.net.name_to_idx(train_comp_identifier)
        return cls(
            surrogate_type=surrogate_type,
            preprocessor_type=preprocessor_type,
            n_components=n_components,
            dataset=dataset,
            # ノード名は「どの種類を置換するか」を指すための入口表記 → ここで種類へ
            # 解決し、以降ノード名は同定情報に残らない。
            comp_type=dataset.net.nodes[train_comp_id].type,
            train_comp_id=train_comp_id,
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
    def train_comp(self) -> Compartment:
        """学習軌道を取った代表ノード。params/初期値の参照先 (置換可否の判定には
        使わない — 種類は comp_type、params 両立基準は replace が持つ)。"""
        return self.dataset.net.nodes[self.train_comp_id]

    @property
    def original_opcost(self) -> OpCost | None:
        """置換前 1 ステップのコスト = 置換対象の種類が持つコスト。"""
        return self.comp_type.opcost

    def simulate(self) -> xr.Dataset:
        return unified_simulator(self.dataset)
