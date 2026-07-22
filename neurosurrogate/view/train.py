"""学習データの可視化: 閉包項に「何を食わせたか」を描く。

学習データの実体は保存されていない — `SurrogateMeta` (dataset/電流/dt) と
`Ansatz.train_source` (どの comp の・先頭何ゲートか) から `bundle.train_xr` を
再生成し、そこから図を組む。→ MLflow から load した run でも同じ図が出る。

evaluate 後の比較図 (specs.py) と違い、**surrogate 単体にしか依存しない** ので
置換シミュを回す前に描ける。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
from matplotlib.figure import Figure

from ..core import access
from ..surrogate.bundle import SurrogateBundle
from .engine import (
    PanelSpec,
    TraceSpec,
    draw_engine,
    error_fig,
    new_figure,
    place_legend,
)

_HIST_BINS = 60
_PANEL_HEIGHT = 1.6  # 時系列図 1 段の高さ [inch]
_FIG_WIDTH = 8.0


def _figsize(n_rows: int) -> tuple[float, float]:
    """段数に応じた寸法。潜在次元やゲート数で段数が変わる図が潰れないように。"""
    return (_FIG_WIDTH, max(4.0, n_rows * _PANEL_HEIGHT))


def _shown(
    bundle: SurrogateBundle, comps: Sequence[int] | None
) -> list[tuple[int, int, str]]:
    """描く学習 comp の (train_source 内の位置, comp_id, 表示名)。comps=None は
    学習 comp 全部。位置は train_inputs / _latents の並び (source.comp_ids 順) と
    対応する。traub19 のような多 comp 学習は全部重ねると読めない → UI で絞る。"""
    nodes = bundle.meta.dataset.net.nodes
    return [
        (k, i, nodes[i].name)
        for k, i in enumerate(bundle.ansatz.train_source(bundle.meta).comp_ids)
        if comps is None or i in comps
    ]


def _latents(bundle: SurrogateBundle, comp_ids: Sequence[int]) -> list[np.ndarray]:
    """comp ごとの潜在軌道 (time, n_components)。閉包項が実際に見た入力。"""
    source = bundle.ansatz.train_source(bundle.meta)
    return [
        np.asarray(bundle.preprocessor.encode(source.gate(bundle.train_xr, i)))
        for i in comp_ids
    ]


def train_raw_fig(
    bundle: SurrogateBundle, comps: Sequence[int] | None = None
) -> Figure:
    """生の学習軌道: 注入電流・学習 comp の V・表示先頭 comp のゲート。

    どの comp の軌道を食わせたかを V パネルで見る。ゲートは表示 comp の先頭 1 個のみ
    (全 comp 分を重ねると本数が comp×gate で潰れる。他 comp のゲートは同一多様体上
    に乗る前提なので、被覆のズレは coverage 図が受け持つ)。
    """
    source = bundle.ansatz.train_source(bundle.meta)
    shown = _shown(bundle, comps)
    return draw_engine(
        [
            PanelSpec("I_ext", [TraceSpec(*access.i_ext(bundle.train_xr))]),
            PanelSpec(
                "V(t) [mV]",
                [
                    TraceSpec(
                        *access.trace(bundle.train_xr, i, access.POTENTIAL_VAR),
                        label=name,
                    )
                    for _, i, name in shown
                ],
            ),
            PanelSpec(
                f"gates ({shown[0][2]})",
                [
                    TraceSpec(
                        access.time(bundle.train_xr),
                        source.gate(bundle.train_xr, shown[0][1])[:, k],
                        label=name,
                    )
                    for k, name in enumerate(
                        bundle.meta.comp_type.gate_names[: source.n_gate]
                    )
                ],
            ),
        ],
        figsize=_figsize(3),
    )


def train_preprocessed_fig(
    bundle: SurrogateBundle, comps: Sequence[int] | None = None
) -> Figure:
    """同定器へ渡す**直前**のデータ (状態列 x と入力列 u を 1 列 1 段、comp 重ね)。

    fit と同じ `ansatz.train_inputs` を呼ぶ → 図に出るのが学習に入ったもの。列構造は
    定式化ごとに違う (sindy=[V, g1..gN] + u / hybrid=[g1..gN] + V) が、view は列名を
    そのまま並べるだけで両方に効く。
    """
    inputs = bundle.ansatz.train_inputs(
        bundle.meta, bundle.train_xr, bundle.preprocessor
    )
    shown = _shown(bundle, comps)
    return draw_engine(
        [
            PanelSpec(
                name,
                [
                    TraceSpec(
                        access.time(bundle.train_xr), mats[pos][:, k], label=label
                    )
                    for pos, _, label in shown
                ],
            )
            for mats, names in ((inputs.x, inputs.x_names), (inputs.u, inputs.u_names))
            for k, name in enumerate(names)
        ],
        figsize=_figsize(len(inputs.x_names) + len(inputs.u_names)),
    )


def train_recon_fig(
    bundle: SurrogateBundle, comps: Sequence[int] | None = None
) -> Figure:
    """preprocessor の再構成誤差 (ゲート → 潜在 → ゲートの RMSE、comp 別)。

    「潜在に落とした時点で何を捨てたか」= 閉包項の同定より手前で決まる誤差の下限。
    """
    source = bundle.ansatz.train_source(bundle.meta)
    shown = _shown(bundle, comps)
    latents = _latents(bundle, [i for _, i, _ in shown])
    return draw_engine(
        [
            PanelSpec(
                "recon RMSE",
                [
                    TraceSpec(
                        access.time(bundle.train_xr),
                        np.sqrt(
                            np.mean(
                                (
                                    source.gate(bundle.train_xr, i)
                                    - np.asarray(
                                        bundle.preprocessor.decode(jnp.asarray(lat))
                                    )
                                )
                                ** 2,
                                axis=1,
                            )
                        ),
                        label=label,
                    )
                    for (_, i, label), lat in zip(shown, latents, strict=True)
                ],
            )
        ]
    )


def train_v_coverage_fig(
    bundle: SurrogateBundle, comps: Sequence[int] | None = None
) -> Figure:
    """学習が踏んだ V の分布 (comp 別ヒストグラム)。

    hybrid の multi-comp 学習は「comp を足して増えるのは V の被覆だけ」を前提に
    している → comp 間で V 分布がどれだけ重なる/ずれるかを見る。評価時にこの範囲を
    外れた電位は外挿になる。
    """
    fig = new_figure()
    ax = fig.subplots()
    for _, i, name in _shown(bundle, comps):
        ax.hist(
            access.potential(bundle.train_xr, i),
            bins=_HIST_BINS,
            histtype="step",
            label=name,
        )
    ax.set_xlabel("V [mV]")
    ax.set_ylabel("count")
    ax.set_title("Training V coverage")
    place_legend(ax)
    return fig


def train_manifold_fig(
    bundle: SurrogateBundle, comps: Sequence[int] | None = None
) -> Figure:
    """潜在空間の軌道 (comp 別)。学習ゲートが乗る多様体の形。

    学習ゲートは params-free なので comp が違っても同一多様体に乗るはず → 軌道が
    重ならなければ multi-comp 学習の前提が崩れている (潜在次元不足か params 混入)。
    """
    shown = _shown(bundle, comps)
    latents = _latents(bundle, [i for _, i, _ in shown])
    latent_names = access.latent_vars(bundle.meta.n_components)
    if bundle.meta.n_components < 2:
        # 潜在が 1 次元なら軌道が描けない → V を横軸に取る (g1 の V 依存を見る)。
        x_label, y_label = access.POTENTIAL_VAR, latent_names[0]
        xs = [access.potential(bundle.train_xr, i) for _, i, _ in shown]
        ys = [lat[:, 0] for lat in latents]
    else:
        x_label, y_label = latent_names[0], latent_names[1]
        xs = [lat[:, 0] for lat in latents]
        ys = [lat[:, 1] for lat in latents]

    fig = new_figure()
    ax = fig.subplots()
    for x, y, (_, _, name) in zip(xs, ys, shown, strict=True):
        ax.plot(x, y, linewidth=0.8, alpha=0.7, label=name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Latent manifold")
    ax.grid(True, linestyle=":", alpha=0.5)
    place_legend(ax)
    return fig


def train_figs(
    bundle: SurrogateBundle, comps: Sequence[int] | None = None
) -> list[tuple[str, Figure]]:
    """学習データ図を識別子付きで一括生成 (specs.draw_all と同じ規約)。
    comps=描く comp の制限 (None=学習 comp 全部)。

    train_xr の再生成はここで初めて走る (cached_property) → 呼ばなければコスト 0。
    """
    jobs: dict[str, Callable[[], Figure]] = {
        "train_raw": lambda: train_raw_fig(bundle, comps),
        "train_preprocessed": lambda: train_preprocessed_fig(bundle, comps),
        "train_recon": lambda: train_recon_fig(bundle, comps),
        "train_v_coverage": lambda: train_v_coverage_fig(bundle, comps),
        "train_manifold": lambda: train_manifold_fig(bundle, comps),
    }
    out: list[tuple[str, Figure]] = []
    for name, job in jobs.items():
        try:
            out.append((name, job()))
        except (ValueError, KeyError, IndexError) as e:
            out.append((name, error_fig(f"{name}: {e}")))
    return out
