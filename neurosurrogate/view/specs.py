from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import xarray as xr
from matplotlib.figure import Figure

from ..core import access
from ..core.access import POTENTIAL_VAR
from .engine import PanelSpec, TraceSpec, draw_engine, error_fig

if TYPE_CHECKING:
    from ..metrics.eval import EvalResult


def spec_simple(ds: xr.Dataset) -> list[PanelSpec]:
    comp_ids = access.comp_ids(ds)
    multi = len(comp_ids) > 1
    spec: list[PanelSpec] = [
        PanelSpec("I_ext", [TraceSpec(*access.i_ext(ds))]),
    ]

    if access.has_i_internal(ds):
        spec.append(
            PanelSpec(
                "I_internal",
                [
                    TraceSpec(*access.i_internal(ds, i), label=f"Comp {i}")
                    for i in comp_ids
                ],
            )
        )

    spec.append(
        PanelSpec(
            "V(t) [mV]",
            [
                TraceSpec(
                    *access.trace(ds, i, POTENTIAL_VAR),
                    label=f"V (Comp {i})" if multi else None,
                )
                for i in comp_ids
            ],
        )
    )

    gate_traces = [
        TraceSpec(*access.trace(ds, i, v), label=f"{v} (Comp {i})")
        for i in comp_ids
        for v in access.gate_variables(ds, i)
    ]
    if gate_traces:
        spec.append(PanelSpec("Gates / Latent", gate_traces))

    return spec


def spec_diff(
    original: xr.Dataset,
    preprocessed: xr.Dataset,
    surrogate: xr.Dataset,
    surr_id: int,
) -> list[PanelSpec]:
    return [
        PanelSpec("I_ext(t)", [TraceSpec(*access.i_ext(original), color="gold")]),
        PanelSpec(
            "V [mV]",
            [
                TraceSpec(
                    *access.trace(original, surr_id, POTENTIAL_VAR),
                    label="orig V",
                    color="blue",
                ),
                TraceSpec(
                    *access.trace(surrogate, surr_id, POTENTIAL_VAR),
                    label="surr V",
                    color="red",
                    style="--",
                ),
            ],
        ),
        *[
            PanelSpec(
                latent,
                [
                    TraceSpec(
                        *access.trace(preprocessed, surr_id, latent),
                        label=f"target {latent}",
                        color="blue",
                    ),
                    TraceSpec(
                        *access.trace(surrogate, surr_id, latent),
                        label=f"surr {latent}",
                        color="red",
                        style="--",
                    ),
                ],
            )
            for latent in access.latent_variables(preprocessed)
        ],
        PanelSpec(
            "orig gates",
            [
                TraceSpec(*access.trace(original, surr_id, name), label=name)
                for name in access.gate_variables(original, surr_id)
            ],
        ),
    ]


def plot_2d_attractor_comparison(orig_ds, surr_ds, comp_id, state_vars=None) -> Figure:
    """相平面重ね描き。orig と surr のダイナミクス一致度可視化。"""
    if state_vars is None:
        state_vars = [access.POTENTIAL_VAR, "latent1"]
    fig = Figure()
    ax = fig.subplots()

    def extract_trajectory(ds):
        # access.trace は (t, y) を返す。相平面は値のみ使う
        return [access.trace(ds, comp_id, var)[1] for var in state_vars]

    # --- 1. データの抽出 ---
    try:
        o_x, o_y = extract_trajectory(orig_ds)
        s_x, s_y = extract_trajectory(surr_ds)
    except KeyError as e:
        ax.text(
            0.5,
            0.5,
            f"Variable not found:\n{e}",
            transform=ax.transAxes,
            ha="center",
            color="red",
        )
        return fig

    # --- 2. 描画 ---
    # オリジナル：黒で「正解」の形を示す。alphaを少し下げて重なりを見やすくする
    ax.plot(
        o_x, o_y, color="black", linewidth=1.2, alpha=0.6, label="Original (Target)"
    )

    # サロゲート：赤（または青）の破線や細線で「再現」を示す
    ax.plot(
        s_x, s_y, color="crimson", linewidth=1.0, alpha=0.8, label="Surrogate (SINDy)"
    )

    # --- 3. 装飾 ---
    ax.set_xlabel(f"{state_vars[0]}")
    ax.set_ylabel(f"{state_vars[1]}")
    ax.set_title(f"Attractor Comparison (Comp {comp_id})")

    # ランダム電流などの場合、軌道がボヤけるのでグリッドがあると位置関係が追いやすい
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()

    return fig


def draw_all(result: EvalResult, comp_id: int) -> list[tuple[str, Figure]]:
    """EvalResult から全描画を識別子付きで一括生成。analysis 側は種別を知らず
    (id, fig) を保存/表示に流すだけ。学習ドメイン外 comp 等での失敗は error_fig
    に畳み戻り値型を保つ。

    latent (preprocessed) は lazy 参照: 学習ドメイン外 comp で preprocessed_latent
    が raise するため diff/attractor でのみ評価する (simple は呼ばない)。
    """
    original, surrogate = result.original_ds, result.surr_ds
    jobs: dict[str, Callable[[], Figure]] = {
        "diff": lambda: draw_engine(
            spec_diff(original, result.preprocessed_latent(comp_id), surrogate, comp_id)
        ),
        "simple": lambda: draw_engine(spec_simple(original)),
        "attractor": lambda: plot_2d_attractor_comparison(
            result.preprocessed_latent(comp_id), surrogate, comp_id
        ),
    }
    out: list[tuple[str, Figure]] = []
    for name, job in jobs.items():
        try:
            out.append((name, job()))
        except (ValueError, KeyError) as e:
            out.append((name, error_fig(f"{name}: {e}")))
    return out
