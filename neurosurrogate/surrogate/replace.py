"""サロゲート置換ドメイン: 学習ドメイン照合による判定と、type 差替の適用。

置換の「誰を (replaceables)・何に (surr_type)・どう差替えるか (replace_nodes)」
を一手に担う。core の NeuronGraph/DatasetConfig は純粋なデータ構造として保ち、
ここが dataclasses.replace で net/nodes だけ差替えて再構築する。
"""

from collections.abc import Callable
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from ..core.network import Compartment, CompartmentType, DatasetConfig, NeuronGraph

if TYPE_CHECKING:
    from .ansatz.base import NeuroSurrogateBase


def resolved_params(comp: Compartment) -> "tuple | None":
    """comp の実効 params: 明示 params、無ければ型 default (param_cls())。

    params 一致判定 (置換ドメイン) の基準。surr のように param_cls=None の型は
    params を持たず None。default は param_cls() が生む NamedTuple デフォルト。
    """
    if comp.params is not None:
        return comp.params
    return comp.type.param_cls() if comp.type.param_cls is not None else None


def replaceable(surrogate: "NeuroSurrogateBase", comp: Compartment) -> bool:
    """comp が surrogate に置換されるか (学習型一致 かつ params 両立)。

    params 両立の基準は surrogate 固有 (surrogate.params_compatible が担う)。
    """
    return comp.type == surrogate.meta.train_comp.type and surrogate.params_compatible(
        comp
    )


def replaceables(surrogate: "NeuroSurrogateBase", dataset: DatasetConfig) -> set[str]:
    """dataset 内の置換対象ノード名を返す (fail first)。

    - 学習型一致 だが params 非両立のノードが1つでもあれば即エラー (疑わしい)
    - 置換対象が皆無なら即エラー (モデルとデータが噛み合わず)
    """
    train = surrogate.meta.train_comp
    targets = {n.name for n in dataset.net.nodes if replaceable(surrogate, n)}

    # 学習型一致 だが未置換 = params 非両立 → 疑わしい (置換不可)
    mismatched = [
        n for n in dataset.net.nodes if n.type == train.type and n.name not in targets
    ]
    if mismatched:
        raise ValueError(
            f"型 {train.type.name!r} 一致だが params 非両立のノード "
            f"{[n.name for n in mismatched]}: 学習ドメイン外。\n"
            f"  train({train.name}): {resolved_params(train)}\n"
            + "\n".join(f"  node({n.name}): {resolved_params(n)}" for n in mismatched)
        )
    if not targets:
        raise ValueError(
            f"学習型 {train.type.name!r} のノードが dataset "
            f"{dataset.model_name!r} に存在しない → 置換対象ゼロ。適用不可"
        )
    return targets


def replaced_names(surrogate: "NeuroSurrogateBase", net: NeuronGraph) -> set[str]:
    """net 内で surrogate が置換するノード名集合を返す (非raise, 診断用)。

    replaceables と違い params 非両立/皆無でも例外を投げず、描画等の情報表示に使う。
    """
    return {n.name for n in net.nodes if replaceable(surrogate, n)}


def replace_nodes(
    net: NeuronGraph,
    new_type: CompartmentType,
    accept: Callable[[Compartment], bool],
) -> NeuronGraph:
    """accept 真ノードの type を new_type に差替 (name/params 保持、構造操作)。"""
    nodes = [dc_replace(n, type=new_type) if accept(n) else n for n in net.nodes]
    return dc_replace(net, nodes=nodes)


def apply_surrogate(
    surrogate: "NeuroSurrogateBase",
    dataset: DatasetConfig,
) -> DatasetConfig:
    """学習ドメインに属す全ノードを surrogate に置換 (検証は replaceables が担う)。"""
    targets = replaceables(surrogate, dataset)
    new_net = replace_nodes(
        dataset.net, surrogate.surr_comp_type, lambda n: n.name in targets
    )
    return dc_replace(dataset, net=new_net)
