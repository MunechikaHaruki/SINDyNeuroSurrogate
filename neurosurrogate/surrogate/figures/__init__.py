"""**surrogate の自己記述図**: 学習済み bundle と適用先ネットだけから描ける図表。

`train.py` (何を食わせたか) と `model.py` (何が出来たか: グラフ・閉包項・
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
from ...plotting import ArtifactEntries, collect
from ..bundle import SurrogateBundle
from ..closure.base import Closure
from ..closure.sindy import SINDyBundle
from ..diagnostics import surrogate_metrics
from ..meta import SurrogateMeta
from ..preprocessor.base import Preprocessor
from ..preprocessor.impl.pca import PCAPreprocessor
from ..replace import replaceable
from .model import neuron_graph_fig, pca_scree_fig, sindy_coef_fig
from .train import (
    train_manifold_fig,
    train_preprocessed_fig,
    train_raw_fig,
    train_recon_fig,
    train_v_coverage_fig,
)


def summary_df(bundles: dict[str, SurrogateBundle]) -> ArtifactEntries:
    """run 軸の学習側指標サマリ (評価結果に依らないので結果無しでも出せる)。"""
    df = pd.DataFrame(
        [{"label": label, **surrogate_metrics(s)} for label, s in bundles.items()]
    ).set_index("label")
    return [("summary", df)]


def closure_figs(closure: Closure) -> ArtifactEntries:
    """閉包項の中身図 (識別子付き)。"""
    if isinstance(closure, SINDyBundle):
        return collect({"model": lambda: sindy_coef_fig(closure)})
    return []


def preprocessor_figs(prep: Preprocessor) -> ArtifactEntries:
    """preprocessor の診断図 (識別子付き)。再構成誤差の時系列は `train_recon_fig`
    が別に受け持つ。"""
    if isinstance(prep, PCAPreprocessor):
        return collect({"pca_scree": lambda: pca_scree_fig(prep)})
    return []


def neuron_graph_figs(
    nets: dict[str, NeuronGraph], meta: SurrogateMeta
) -> ArtifactEntries:
    """適用先ごとのニューロングラフ (識別子 `<target>/neurograph`)。強調するノードは
    meta との置換可否から引く = 呼び出し側は「どの適用先を描くか」だけ渡す
    (描画なので置換不可 = 強調ゼロでも例外にしない。検証は `replaceables` の関心)。"""
    return collect(
        {
            f"{target}/neurograph": partial(
                neuron_graph_fig,
                net,
                {n.name for n in net.nodes if replaceable(meta, n)},
            )
            for target, net in nets.items()
        }
    )


def train_figs(
    bundle: SurrogateBundle,
    comps: Sequence[int] | None = None,
    i_ext_ylim: tuple[float, float] | None = None,
) -> ArtifactEntries:
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
