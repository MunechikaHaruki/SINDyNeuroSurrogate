"""**学習 run 1 本について描ける図**。置換シミュの結果を受け取らない
= `SeriesView` を import しないのがこの module の不変条件 (結果が要る図は
`series` / `report` の担当)。marimo/MLflow 非依存。

描く中身は run 自身が決める (`surrogate.figures.surrogate_figs` が bundle の型から
解く) = 「何を描くか」の宣言をここが持たない。
"""

from __future__ import annotations

from ..plotting import Artifact, use_style
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.figures import surrogate_figs


def model_figs(bundle: SurrogateBundle, view_comps: tuple[str, ...]) -> list[Artifact]:
    """surrogate **1 本**の自己記述図。

    **run 横断のサマリ表はここに無い** — 中身が「今 何本を比べているか」で変わるので
    レポートの産物 (レポートに属さないと別の選択で描き替わる)。

    描く対象は学習 run そのもの (置換シミュの結果を受け取らない)。適用先も bundle の
    学習 dataset から解くため、系列や report run を必要としない。**1 本ずつ返す** =
    run 軸で回すのは run_id を段の名前へ解ける呼び出し側の関心。
    """
    use_style()
    net = bundle.meta.dataset.net
    return surrogate_figs(bundle, net, [net.name_to_idx(c) for c in view_comps] or None)
