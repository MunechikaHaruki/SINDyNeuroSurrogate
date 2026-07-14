"""サロゲート置換ドメイン: 学習ドメイン照合による判定と、type 差替の適用。

置換の「誰を (replaceables)・何に (surr_type)・どう差替えるか (replace_nodes)」
を一手に担う。core の NeuronGraph/DatasetConfig は純粋なデータ構造として保ち、
ここが dataclasses.replace で net/nodes だけ差替えて再構築する。
"""

from collections.abc import Callable
from dataclasses import replace as dc_replace
from enum import Enum, auto
from typing import TYPE_CHECKING

from ..core.network import Compartment, CompartmentType, DatasetConfig, NeuronGraph

if TYPE_CHECKING:
    from .base import NeuroSurrogateBase


class Verdict(Enum):
    """サロゲート置換の妥当性判定 (学習ドメインとの照合結果)。"""

    REPLACE = auto()  # 型一致 かつ params 両立 → 置換
    MISMATCH = auto()  # 型一致 だが params 非両立 → 疑わしい (置換不可)
    SKIP = auto()  # 型不一致 → 無関係 (対象外)


def verdict(surrogate: "NeuroSurrogateBase", comp: Compartment) -> Verdict:
    """comp が surrogate の学習ドメイン (型 + params 両立性) に属すか判定。

    型が違えば無関係 (SKIP)、型は同じだが params が非両立なら疑わしい
    (MISMATCH)、両方 OK で置換可 (REPLACE)。params 両立性の基準は
    surrogate 固有 (surrogate.params_compatible が担う)。
    """
    if comp.type != surrogate.meta.train_comp.type:
        return Verdict.SKIP
    if not surrogate.params_compatible(comp):
        return Verdict.MISMATCH
    return Verdict.REPLACE


def replaceables(surrogate: "NeuroSurrogateBase", dataset: DatasetConfig) -> set[str]:
    """dataset 内の置換対象ノード名を返す (fail first)。

    - 型一致・params 非両立 (MISMATCH) が1つでもあれば即エラー
    - 置換対象 (REPLACE) が皆無なら即エラー (モデルとデータが噛み合わず)
    """
    verdicts = {n.name: verdict(surrogate, n) for n in dataset.net.nodes}
    train = surrogate.meta.train_comp

    mismatched = [n for n in dataset.net.nodes if verdicts[n.name] is Verdict.MISMATCH]
    if mismatched:
        raise ValueError(
            f"型 {train.type.name!r} 一致だが params 非両立のノード "
            f"{[n.name for n in mismatched]}: 学習ドメイン外。\n"
            f"  train({train.name}): {train.resolved_params}\n"
            + "\n".join(f"  node({n.name}): {n.resolved_params}" for n in mismatched)
        )
    targets = {name for name, v in verdicts.items() if v is Verdict.REPLACE}
    if not targets:
        raise ValueError(
            f"学習型 {train.type.name!r} のノードが dataset "
            f"{dataset.model_name!r} に存在しない → 置換対象ゼロ。適用不可"
        )
    return targets


def replace_nodes(
    net: NeuronGraph,
    new_type: CompartmentType,
    accept: Callable[[Compartment], bool],
) -> NeuronGraph:
    """accept 真ノードの type を new_type に差替 (name/params 保持、構造操作)。"""
    nodes = [dc_replace(n, type=new_type) if accept(n) else n for n in net.nodes]
    return dc_replace(net, nodes=nodes)


def apply(
    surrogate: "NeuroSurrogateBase",
    dataset: DatasetConfig,
) -> DatasetConfig:
    """学習ドメインに属す全ノードを surrogate に置換 (検証は replaceables が担う)。"""
    targets = replaceables(surrogate, dataset)
    new_net = replace_nodes(
        dataset.net, surrogate.surr_comp_type, lambda n: n.name in targets
    )
    return dc_replace(dataset, net=new_net)
