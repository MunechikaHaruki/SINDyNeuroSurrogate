"""サロゲート置換ドメイン: 学習ドメイン照合による判定と、type 差替の適用。

置換の「誰を (replaceables)・何に (surr_type)・どう差替えるか (replace_nodes)」
を一手に担う。core の NeuronGraph/DatasetConfig は純粋なデータ構造として保ち、
ここが dataclasses.replace で net/nodes だけ差替えて再構築する。
"""

from collections.abc import Callable
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from ..core.network import Compartment, CompartmentType, DatasetConfig, NeuronGraph
from .meta import SurrogateMeta

if TYPE_CHECKING:
    from .bundle import SurrogateBundle


# ---------------------------------------------------------------------------
# params 両立基準は surrogate_type ごとに違う: 学習モデルがどの物理 params を自分の
# 中へ焼き込むかで決まる。方程式の定式化 (ansatz) でなく置換ドメインの関心事なので
# ここに集約する。
# ---------------------------------------------------------------------------
_PARAMS_MATCH: dict[str, Callable[[tuple | None, tuple | None], bool]] = {
    # sindy: surr は param_cls=None → simulator がノード params を捨て、学習モデルが
    # V+gate 全体を train params 込みで再現する → 全 params 完全一致が必須。
    "sindy": lambda train, node: train == node,
    # hybrid: 物理 dv も Ca サブ系も置換先ノード自身の params で解く → params 自由
    # (型一致のみで置換可)。
    "hybrid": lambda train, node: True,
}


def replaceable(meta: SurrogateMeta, comp: Compartment) -> bool:
    """comp が surrogate に置換されるか (**種類一致** かつ params 両立)。

    サロゲートは「コンパートメントの種類 → それを置換するモデル」の対応 → 種類は
    meta.comp_type と直接照合する (学習元ノード経由で導出しない)。params 両立だけが
    学習元ノードとの比較になる (sindy は学習時 params を焼き込むため)。
    """
    if comp.type != meta.comp_type:
        return False
    return _PARAMS_MATCH[meta.surrogate_type](
        meta.train_comp.resolved_params, comp.resolved_params
    )


def replaceables(meta: SurrogateMeta, dataset: DatasetConfig) -> set[str]:
    """dataset 内の置換対象ノード名を返す (fail first)。

    - 種類一致 だが params 非両立のノードが1つでもあれば即エラー (疑わしい)
    - 置換対象が皆無なら即エラー (モデルとデータが噛み合わず)
    """
    targets = {n.name for n in dataset.net.nodes if replaceable(meta, n)}

    # 種類一致 だが未置換 = params 非両立 → 疑わしい (置換不可)
    mismatched = [
        n
        for n in dataset.net.nodes
        if n.type == meta.comp_type and n.name not in targets
    ]
    if mismatched:
        train = meta.train_comp
        raise ValueError(
            f"種類 {meta.comp_type.name!r} 一致だが params 非両立のノード "
            f"{[n.name for n in mismatched]}: 学習ドメイン外。\n"
            f"  train({train.name}): {train.resolved_params}\n"
            + "\n".join(f"  node({n.name}): {n.resolved_params}" for n in mismatched)
        )
    if not targets:
        raise ValueError(
            f"種類 {meta.comp_type.name!r} のノードが dataset "
            f"{dataset.model_name!r} に存在しない → 置換対象ゼロ。適用不可"
        )
    return targets


def replaced_names(meta: SurrogateMeta, net: NeuronGraph) -> set[str]:
    """net 内で surrogate が置換するノード名集合を返す (非raise, 診断用)。

    replaceables と違い params 非両立/皆無でも例外を投げず、描画等の情報表示に使う。
    """
    return {n.name for n in net.nodes if replaceable(meta, n)}


def replace_nodes(
    net: NeuronGraph,
    new_type: CompartmentType,
    accept: Callable[[Compartment], bool],
) -> NeuronGraph:
    """accept 真ノードの type を new_type に差替 (name/params 保持、構造操作)。"""
    nodes = [dc_replace(n, type=new_type) if accept(n) else n for n in net.nodes]
    return dc_replace(net, nodes=nodes)


def apply_surrogate(
    bundle: "SurrogateBundle",
    dataset: DatasetConfig,
) -> DatasetConfig:
    """学習ドメインに属す全ノードを surrogate に置換 (検証は replaceables が担う)。

    判定 3 関数と違い meta だけでは足りない (差替える型 = 学習成果物から組む
    `surr_comp_type` が要る) ので、ここだけ bundle を受ける。
    """
    targets = replaceables(bundle.meta, dataset)
    new_net = replace_nodes(
        dataset.net, bundle.surr_comp_type, lambda n: n.name in targets
    )
    return dc_replace(dataset, net=new_net)
