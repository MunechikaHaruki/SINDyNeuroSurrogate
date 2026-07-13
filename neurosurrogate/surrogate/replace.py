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
    from .base import SurrogateMeta


class Verdict(Enum):
    """サロゲート置換の妥当性判定 (学習ドメインとの照合結果)。"""

    REPLACE = auto()  # 型一致 かつ params 一致 → 置換
    MISMATCH = auto()  # 型一致 だが params 不一致 → 疑わしい (置換不可)
    SKIP = auto()  # 型不一致 → 無関係 (対象外)


def verdict(meta: "SurrogateMeta", comp: Compartment) -> Verdict:
    """comp が学習ドメイン (train_comp の type+params) に属すか判定。

    型が違えば無関係 (SKIP)、型は同じだが params が違えば疑わしい
    (MISMATCH)、両方一致で置換可 (REPLACE)。
    """
    train = meta.train_comp
    if comp.type != train.type:
        return Verdict.SKIP
    if comp.params != train.params:
        return Verdict.MISMATCH
    return Verdict.REPLACE


def replaceables(meta: "SurrogateMeta", dataset: DatasetConfig) -> set[str]:
    """dataset 内の置換対象ノード名を返す (fail first)。

    - 型一致・params 不一致 (MISMATCH) が1つでもあれば即エラー
    - 置換対象 (REPLACE) が皆無なら即エラー (モデルとデータが噛み合わず)
    """
    verdicts = {n.name: verdict(meta, n) for n in dataset.net.nodes}
    train = meta.train_comp

    mismatched = [n for n in dataset.net.nodes if verdicts[n.name] is Verdict.MISMATCH]
    if mismatched:
        raise ValueError(
            f"型 {train.type.name!r} 一致だが params 不一致のノード "
            f"{[n.name for n in mismatched]}: サロゲートは学習 params 専用。\n"
            f"  train({train.name}): {train.params}\n"
            + "\n".join(f"  node({n.name}): {n.params}" for n in mismatched)
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
    meta: "SurrogateMeta",
    surr_type: CompartmentType,
    dataset: DatasetConfig,
) -> DatasetConfig:
    """学習ドメインに属す全ノードを surrogate に置換 (検証は replaceables が担う)。"""
    targets = replaceables(meta, dataset)
    new_net = replace_nodes(dataset.net, surr_type, lambda n: n.name in targets)
    return dc_replace(dataset, net=new_net)
