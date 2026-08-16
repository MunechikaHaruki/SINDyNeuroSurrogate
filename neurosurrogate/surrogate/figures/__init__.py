"""**surrogate の自己記述図**: 学習済み bundle と適用先ネットだけから描ける図表。

`_train.py` (何を食わせたか) と `model.py` (何が出来たか: グラフ・閉包項・
preprocessor) の 2 本。**評価結果を一切受け取らない**のが境界 — 置換シミュを回す前
に描けるものだけがここに居る (原系との突き合わせは `neurosurrogate.waveform`)。

`closure_figs`/`preprocessor_figs` は共通の図が無い (SINDy=ξ heatmap、PCA=scree、
AE や NN 表現は固有図なし) → 型で振り分け、図を持たない表現は空列を返す。
marimo/mlflow 非依存。
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import pandas as pd

from ...core.network import NeuronGraph
from ...plotting import Artifact, collect, use_style
from ..bundle import SurrogateBundle
from ..closure.base import Closure
from ..closure.sindy import SINDyBundle
from ..diagnostics import surrogate_metrics
from ..meta import SurrogateMeta
from ..preprocessor.base import Preprocessor
from ..preprocessor.impl.pca import PCAPreprocessor
from ..replace import replaceable
from ._train import (
    train_manifold_fig,
    train_preprocessed_fig,
    train_raw_fig,
    train_recon_fig,
    train_v_coverage_fig,
)
from .model import neuron_graph_fig, pca_scree_fig, sindy_coef_fig


def summary_df(bundles: dict[str, SurrogateBundle]) -> list[Artifact]:
    """run 軸の学習側指標サマリ (評価結果に依らないので結果無しでも出せる)。"""
    df = pd.DataFrame(
        [{"label": label, **surrogate_metrics(s)} for label, s in bundles.items()]
    ).set_index("label")
    return [Artifact("summary", df)]


def closure_figs(closure: Closure) -> list[Artifact]:
    """閉包項の中身図 (識別子付き)。"""
    if isinstance(closure, SINDyBundle):
        return collect({"model": lambda: sindy_coef_fig(closure)})
    return []


def preprocessor_figs(prep: Preprocessor) -> list[Artifact]:
    """preprocessor の診断図 (識別子付き)。再構成誤差の時系列は `train_recon_fig`
    が別に受け持つ。"""
    if isinstance(prep, PCAPreprocessor):
        return collect({"pca_scree": lambda: pca_scree_fig(prep)})
    return []


def neuron_graph_figs(net: NeuronGraph, meta: SurrogateMeta) -> list[Artifact]:
    """適用先のニューロングラフ (識別子 `neurograph`)。強調するノードは meta との
    置換可否から引く = 呼び出し側は適用先ネットだけ渡す (描画なので置換不可 =
    強調ゼロでも例外にしない。検証は `replaceables` の関心)。"""
    return collect(
        {
            "neurograph": partial(
                neuron_graph_fig,
                net,
                {n.name for n in net.nodes if replaceable(meta, n)},
            )
        }
    )


def train_figs(
    bundle: SurrogateBundle,
    comps: Sequence[int] | None = None,
    i_ext_ylim: tuple[float, float] | None = None,
) -> list[Artifact]:
    """学習データ図を識別子付きで一括生成。comps=描く comp の制限 (None=学習 comp
    全部)。i_ext_ylim=diff.png と軸を揃えたいとき渡す (発表用)。

    train_xr の再生成はここで初めて走る (cached_property) → 呼ばなければコスト 0。
    """
    return collect(
        {
            "train_raw": lambda: train_raw_fig(bundle, comps, i_ext_ylim),
            "train_preprocessed": lambda: train_preprocessed_fig(bundle, comps),
            "train_recon": lambda: train_recon_fig(bundle, comps),
            "train_v_coverage": lambda: train_v_coverage_fig(bundle, comps),
            "train_manifold": lambda: train_manifold_fig(bundle, comps),
        }
    )


def surrogate_figs(
    bundle: SurrogateBundle, view_comps: tuple[str, ...] = ()
) -> list[Artifact]:
    """**run 1 本が自分について描けるもの全部** (置換シミュの結果を受け取らない)。
    run_id を渡せば図が出てくる、の実体で、**1 本ずつ返す** = run 軸で回すのは
    run_id を段の名前へ解ける呼び出し側の関心。

    何を描くかは宣言で選ばず bundle の中身が決める (SINDy なら ξ heatmap、PCA なら
    scree、固有図を持たない表現は何も出さない) → 種類の一覧を持つ層がどこにも要らない。
    適用先も学習 dataset から解く (`view_comps` は描く comp を名前で絞る指定だけ) =
    系列も評価 run も要らない。

    **run 横断のサマリ表 (`summary_df`) はここに含めない** — 中身が「今 何本を比べて
    いるか」で変わるのでレポートの産物 (`sim.report.report`)。
    """
    net = bundle.meta.dataset.net
    use_style()
    return [
        *closure_figs(bundle.closure),
        *preprocessor_figs(bundle.preprocessor),
        *neuron_graph_figs(net, bundle.meta),
        *train_figs(bundle, [net.name_to_idx(c) for c in view_comps] or None),
    ]
