"""外から使う成果物生成関数の集約 (Figure/DataFrame を返す関数のみ)。呼び出し側
(`metrics/report.py`) はここだけを見れば済み、submodule 単位の分割は内部詳細。

**複数の個別図/指標をまとめて返す集約関数** (`cell_figs`/`closure_figs`/
`preprocessor_figs`/`neuron_graph_figs`/`train_figs`/`wave_report`) の実装は
ここに置く。個別の図生成/指標計算 (`panels_simple`/`_sindy_coef_fig`/
`train_raw_fig`/`waveform_summary_df` 等) は各 submodule
(`cell.py`/`model.py`/`train.py`/`wave_table.py`) に残す。

計算層 (`wave.py`: `DynamicMetrics` とスカラー/tuple を返す純粋関数群) は
Figure/DataFrame を返さないのでここでは公開しない (必要なら `.wave` を直接 import
する)。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from ...core.network import NeuronGraph
from ...surrogate.bundle import SurrogateBundle
from ...surrogate.closure.base import Closure
from ...surrogate.closure.sindy import SINDyBundle
from ...surrogate.diagnostics import surrogate_metrics
from ...surrogate.meta import SurrogateMeta
from ...surrogate.preprocessor.base import Preprocessor
from ...surrogate.preprocessor.impl.pca import PCAPreprocessor
from ...surrogate.replace import replaced_names
from ._internal.engine import collect, draw_engine, error_fig
from ._internal.wave import DynamicMetrics, n_spikes, spike_shape_corr, waveform_summary
from .cell import attractor_fig, current_preview_fig, panels_diff, panels_simple
from .grid import compare_grid_fig, metric_fig, trace_grid_fig
from .model import _sindy_coef_fig, equation_texs, neuron_graph_fig, pca_scree_fig
from .train import (
    train_manifold_fig,
    train_preprocessed_fig,
    train_raw_fig,
    train_recon_fig,
    train_v_coverage_fig,
)
from .wave_table import spike_features_df, waveform_summary_df

__all__ = [
    "ArtifactEntries",
    "KIND_FUNCS",
    "cell_figs",
    "closure_figs",
    "compare_grid_fig",
    "current_preview_fig",
    "equation_texs",
    "error_fig",
    "metric_fig",
    "neuron_graph_figs",
    "preprocessor_figs",
    "summary_df",
    "trace_grid_fig",
    "train_figs",
    "wave_report",
]

# 集約関数が返す (識別子, 成果物) 列の要素型。今は Figure しか返さないが、
# 将来 DataFrame を返す集約関数が増えても型で表現できるようにしておく。
# list は invariant で `collect` の list[tuple[str, Figure]] を受け付けないため
# covariant な Sequence にする。
ArtifactEntries = Sequence[tuple[str, Figure | pd.DataFrame]]


def summary_df(bundles: dict[str, SurrogateBundle]) -> pd.DataFrame:
    """run 軸の学習側指標サマリ (評価結果に依らないので results 無しでも出せる)。"""
    return pd.DataFrame(
        [{"label": label, **surrogate_metrics(s)} for label, s in bundles.items()]
    ).set_index("label")


def cell_figs(
    original: xr.Dataset,
    surrogate: xr.Dataset,
    comp_id: int,
    latent: Callable[[], xr.Dataset],
    comps: Sequence[int] | None = None,
) -> ArtifactEntries:
    """1 セルの全描画を識別子付きで一括生成 (失敗の畳み込みは `collect`)。
    呼び出し側は種別を知らず (id, fig) を保存/表示に流すだけ。

    comp_id=比較対象 (diff/attractor は 1 comp の話)、comps=全 comp を並べる図
    (simple) の表示制限。

    `latent` (原系ゲートの潜在射影) は **callable で受けて lazy 参照**: 学習ドメイン
    外 comp では raise するので diff/attractor でのみ評価する (simple は呼ばない)。
    """
    return collect(
        {
            "diff": lambda: draw_engine(
                panels_diff(original, latent(), surrogate, comp_id)
            ),
            "simple": lambda: draw_engine(panels_simple(original, comps)),
            "attractor": lambda: attractor_fig(latent(), surrogate, comp_id),
        }
    )


def closure_figs(closure: Closure) -> ArtifactEntries:
    """閉包項の中身図 (識別子付き)。中身の描き方は表現ごとに違い、共通の図は無い
    (SINDy=ξ heatmap、NN 表現なら重み分布など) → 型で振り分ける。図を持たない表現
    は空列を返し、呼び出し側は保存/表示に流すだけで済む。"""
    if isinstance(closure, SINDyBundle):
        return collect({"model": lambda: _sindy_coef_fig(closure)})
    return []


def preprocessor_figs(prep: Preprocessor) -> ArtifactEntries:
    """preprocessor の診断図 (識別子付き)。指標の見せ方は変換ごとに違い共通図が無い
    (PCA=寄与率 scree、AE は固有図なし) → closure_figs と同型で型振り分け。図を持た
    ない変換は空列を返す。再構成誤差の時系列は train_recon_fig が別に受け持つ。"""
    if isinstance(prep, PCAPreprocessor):
        return collect({"pca_scree": lambda: pca_scree_fig(prep)})
    return []


def neuron_graph_figs(
    nets: dict[str, NeuronGraph], meta: SurrogateMeta
) -> ArtifactEntries:
    """適用先ごとのニューロングラフ (識別子 `<target>/neurograph`)。置換ノードの強調は
    meta から引く = 呼び出し側は「どの適用先を描くか」だけ渡す。"""
    return collect(
        {
            f"{target}/neurograph": partial(
                neuron_graph_fig, net, replaced_names(meta, net)
            )
            for target, net in nets.items()
        }
    )


def train_figs(
    bundle: SurrogateBundle, comps: Sequence[int] | None = None
) -> ArtifactEntries:
    """学習データ図を識別子付きで一括生成 (`sim.draw_all` と同じ `collect` 規約)。
    comps=描く comp の制限 (None=学習 comp 全部)。

    train_xr の再生成はここで初めて走る (cached_property) → 呼ばなければコスト 0。
    """
    return collect(
        {
            "train_raw": lambda: train_raw_fig(bundle, comps),
            "train_preprocessed": lambda: train_preprocessed_fig(bundle, comps),
            "train_recon": lambda: train_recon_fig(bundle, comps),
            "train_v_coverage": lambda: train_v_coverage_fig(bundle, comps),
            "train_manifold": lambda: train_manifold_fig(bundle, comps),
        }
    )


@dataclass(frozen=True)
class WaveReport:
    """波形+スパイク指標を統合した評価レポート。df をそのまま表示/保存へ流す。"""

    df_metrics: pd.DataFrame  # 波形行 (+ 指定 spike が両信号にあればその特徴量)
    df_scalar: pd.DataFrame  # 全スカラーを縦持ち


def wave_report(
    dm: DynamicMetrics,
    spike_orig: int = 0,
    spike_surr: int = 0,
) -> WaveReport:
    """dm から波形/スパイク指標を計算し DataFrame まで組み立てて返す。指定した
    spike index が両信号の範囲内にあるときだけ、その AP の特徴量と形状相関を足す。"""
    n_orig, n_surr = n_spikes(dm)
    df_metrics = waveform_summary_df(dm)
    scalar = waveform_summary(dm)
    if 0 <= spike_orig < n_orig and 0 <= spike_surr < n_surr:
        df_spike = spike_features_df(dm, spike_orig=spike_orig, spike_surr=spike_surr)
        df_spike.index.name = "metric"
        df_metrics = pd.concat([df_metrics, df_spike])
        scalar.update(spike_shape_corr(dm))
    return WaveReport(
        df_metrics=df_metrics,
        df_scalar=pd.DataFrame(scalar.items(), columns=["metric", "value"]).set_index(
            "metric"
        ),
    )


# `declare.ALL_KINDS` の単一源。関数名がそのまま `draw.json` の `kinds` キーになる
# ので、ここへ関数を並べておけば rename が自動で追従する (文字列を手で書き写して
# ズレる余地を無くす)。`cell` 系だけは `cell_figs` の呼び出しに付随して
# `wave_report` も呼ぶ複合キーだが、キー名自体は `cell_figs` で足りる。
KIND_FUNCS = (
    current_preview_fig,
    summary_df,
    closure_figs,
    preprocessor_figs,
    neuron_graph_figs,
    train_figs,
    trace_grid_fig,
    cell_figs,
    metric_fig,
    compare_grid_fig,
)
