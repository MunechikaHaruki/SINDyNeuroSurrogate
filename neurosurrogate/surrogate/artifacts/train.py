"""学習データ成果物: 閉包項に「何を食わせたか」を描く。

学習データの実体は保存されていない — `SurrogateMeta` (dataset/電流/dt) と
`Ansatz.train_source` (どの comp の・先頭何ゲートか) から `bundle.train_xr` を
再生成し、そこから図を組む。→ MLflow から load した run でも同じ図が出る。

evaluate 後の比較図 (sim.py) と違い、**surrogate 単体にしか依存しない** ので
置換シミュを回す前に描ける。
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from ...artifact.model import Artifact
from ...artifact.plotting import (
    PanelSpec,
    TraceSpec,
    draw_engine,
    new_figure,
    place_legend,
)
from ...core import access
from ..bundle import SurrogateBundle

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


def _soma_ids(bundle: SurrogateBundle) -> list[int]:
    """soma comp_id 一覧。全モデル共通で "soma" 固定命名 (neurons/__init__.py)。"""
    nodes = bundle.meta.dataset.net.nodes
    return [
        i
        for i in bundle.ansatz.train_source(bundle.meta).comp_ids
        if nodes[i].name == "soma"
    ]


def _latents(bundle: SurrogateBundle, comp_ids: Sequence[int]) -> list[np.ndarray]:
    """comp ごとの潜在軌道 (time, n_components)。閉包項が実際に見た入力。"""
    source = bundle.ansatz.train_source(bundle.meta)
    return [
        np.asarray(bundle.preprocessor.encode(source.gate(bundle.train_xr, i)))
        for i in comp_ids
    ]


def train_raw_artifact(
    bundle: SurrogateBundle,
    comps: Sequence[int] | None = None,
    i_ext_ylim: tuple[float, float] | None = None,
) -> Artifact:
    """生の学習軌道: 注入電流・学習 comp の V・表示先頭 comp のゲート。

    どの comp の軌道を食わせたかを V パネルで見る。ゲートは表示 comp の先頭 1 個のみ
    (全 comp 分を重ねると本数が comp×gate で潰れる。他 comp のゲートは同一多様体上
    に乗る前提なので、被覆のズレは coverage 図が受け持つ)。i_ext_ylim は diff.png の
    I_ext(t) と軸を揃えたいとき呼び出し側から渡す (発表用)。
    """
    source = bundle.ansatz.train_source(bundle.meta)
    shown = _shown(bundle, _soma_ids(bundle))
    return Artifact(
        "train_raw",
        draw_engine(
            [
                PanelSpec(
                    "I_ext(t)\n[μA/cm²]",
                    [TraceSpec(*access.i_ext(bundle.train_xr), color="#FFC107")],
                    ylim=i_ext_ylim,
                ),
                PanelSpec(
                    "v(t) [mV]",
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
                            # 表記はポスター本文 (m, n, h, ...) に揃える
                            label=name.lower(),
                        )
                        for k, name in enumerate(
                            bundle.meta.comp_type.gate_names[: source.n_gate]
                        )
                    ],
                ),
            ],
            figsize=_figsize(3),
        ),
    )


def train_preprocessed_artifact(
    bundle: SurrogateBundle, comps: Sequence[int] | None = None
) -> Artifact:
    """同定器へ渡す**直前**の圧縮済みデータ (状態列 x を 1 列 1 段、comp 重ね)。

    fit と同じ `ansatz.train_inputs` を呼ぶ → 図に出るのが学習に入ったもの。V は圧縮
    対象でない (hybrid では入力 u、sindy では x の 1 列として素通し) ので、圧縮後の図
    には出さない。
    """
    inputs = bundle.ansatz.train_inputs(
        bundle.meta, bundle.train_xr, bundle.preprocessor
    )
    shown = _shown(bundle, _soma_ids(bundle))
    panels = [
        PanelSpec(
            name,
            [
                TraceSpec(
                    access.time(bundle.train_xr),
                    mats[pos][:, k],
                    label=label,
                    color="red",
                )
                for pos, _, label in shown
            ],
        )
        for mats, names in ((inputs.x, inputs.x_names),)
        for k, name in enumerate(names)
        if name != access.POTENTIAL_VAR  # V は圧縮していない → 圧縮後の図には出さない
    ]
    return Artifact("train_preprocessed", draw_engine(panels, figsize=_figsize(3)))


def train_recon_artifact(
    bundle: SurrogateBundle, comps: Sequence[int] | None = None
) -> Artifact:
    """preprocessor の再構成誤差 (ゲート → 潜在 → ゲートの RMSE、comp 別)。

    「潜在に落とした時点で何を捨てたか」= 閉包項の同定より手前で決まる誤差の下限。
    """
    source = bundle.ansatz.train_source(bundle.meta)
    shown = _shown(bundle, comps)
    latents = _latents(bundle, [i for _, i, _ in shown])
    return Artifact(
        "train_recon",
        draw_engine(
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
        ),
    )


def train_v_coverage_artifact(
    bundle: SurrogateBundle, comps: Sequence[int] | None = None
) -> Artifact:
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
    ax.set_xlabel("v [mV]")
    ax.set_ylabel("count")
    ax.set_title("Training V coverage")
    place_legend(ax)
    return Artifact("train_v_coverage", fig)


def train_manifold_artifact(
    bundle: SurrogateBundle, comps: Sequence[int] | None = None
) -> Artifact:
    """潜在空間の軌道 (comp 別)。学習ゲートが乗る多様体の形。

    学習ゲートは params-free なので comp が違っても同一多様体に乗るはず → 軌道が
    重ならなければ multi-comp 学習の前提が崩れている (潜在次元不足か params 混入)。
    """
    shown = _shown(bundle, comps)
    latents = _latents(bundle, [i for _, i, _ in shown])
    latent_names = access.latent_vars(bundle.meta.n_components)
    if bundle.meta.n_components < 2:
        # 潜在が 1 次元なら軌道が描けない → V を横軸に取る (z1 の V 依存を見る)。
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
    return Artifact("train_manifold", fig)
