"""評価/描画側から呼ばれる surrogate 診断関数群。marimo/mlflow 非依存。

`SurrogateBundle` は setup/load/save + ansatz 委譲だけを持つ本体 → 評価・描画が
要求する診断 (学習ドメイン外チェック込みの変換など) はここへ集める。bundle を
肥大化させない置き場所。
"""

from typing import TYPE_CHECKING

import xarray as xr

from ..core.coords import transform_gate
from ..core.network import NeuronGraph
from .replace import replaceable

if TYPE_CHECKING:
    from .bundle import SurrogateBundle


def surrogate_metrics(bundle: "SurrogateBundle") -> dict:
    """MLflow へ流す学習側指標サマリ。closure/preprocessor 固有指標 + 演算コスト
    差分 (`cost/*`)。**cost/* のキー空間組立はここの関心** (OpCost 代数側は持たない)。
    surr のコストは `surr_comp_type` に焼き込み済 (別経路を持たない)。original が
    無ければ差分は出さない。"""
    orig = bundle.meta.original_opcost
    cost: dict[str, int] = {}
    if orig is not None:
        surr = bundle.surr_comp_type.opcost
        assert surr is not None  # surr_comp_type は必ず opcost を焼き込む
        surr_d = surr.to_dict()
        orig_d = orig.to_dict()
        cost = {
            **{f"cost/surrogate/{k}": v for k, v in surr_d.items()},
            **{f"cost/original/{k}": v for k, v in orig_d.items()},
            **{f"cost/surr-orig/{k}": surr_d[k] - orig_d[k] for k in orig_d},
        }
    return {
        **bundle.closure.metrics(),
        **bundle.preprocessor.metrics(),
        **cost,
    }


def preprocessed_latent(
    surrogate: "SurrogateBundle", net: NeuronGraph, ds: xr.Dataset, comp_id: int
) -> xr.Dataset:
    """comp_id ノードの原系ゲートを surrogate の latent 空間へ射影した (V, latent...)
    xr (診断用)。置換対象外 (学習ドメイン外) は latent 比較不可。
    """
    comp = net.nodes[comp_id]
    if not replaceable(surrogate.meta, comp):
        raise ValueError(
            f"comp {comp.name!r} is outside the trained domain -> latent comparison "
            f"not possible (trained type {surrogate.meta.comp_type.name!r})"
        )
    return transform_gate(surrogate.preprocessor, ds, comp_id)
