from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from .profiler_view import PanelSpec, TraceSpec, draw_engine


def spec_simple(ds: xr.Dataset) -> list[PanelSpec]:
    comp_ids = np.unique(ds.coords["comp_id"].values)
    multi = len(comp_ids) > 1
    spec: list[PanelSpec] = [
        PanelSpec("I_ext", TraceSpec(ds["I_ext"])),
    ]

    if "I_internal" in ds:
        spec.append(
            PanelSpec(
                "I_internal",
                [
                    TraceSpec(ds["I_internal"].sel(node_id=i), label=f"Comp {i}")
                    for i in comp_ids
                ],
            )
        )

    spec.append(
        PanelSpec(
            "V(t) [mV]",
            [
                TraceSpec(
                    ds["vars"].sel(gate=False, comp_id=i),
                    label=f"V (Comp {i})" if multi else None,
                )
                for i in comp_ids
            ],
        )
    )

    gate_traces = [
        TraceSpec(
            ds["vars"].sel(gate=True, comp_id=i, variable=v),
            label=f"{v} (Comp {i})",
        )
        for i in comp_ids
        for v in np.unique(
            ds["vars"].sel(gate=True, comp_id=i).coords["variable"].values
        )
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
    v_sel = {"comp_id": surr_id, "variable": "V"}
    latent_vars = [v for v in preprocessed.coords["variable"].values if v != "V"]
    gate_data = original["vars"].sel(comp_id=surr_id, gate=True)

    return [
        PanelSpec("I_ext(t)", TraceSpec(original["I_ext"], color="gold")),
        PanelSpec(
            "V [mV]",
            [
                TraceSpec(original["vars"].sel(**v_sel), label="orig V", color="blue"),
                TraceSpec(
                    surrogate["vars"].sel(**v_sel),
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
                        preprocessed["vars"].sel(variable=latent),
                        label=f"target {latent}",
                        color="blue",
                    ),
                    TraceSpec(
                        surrogate["vars"].sel(comp_id=surr_id, variable=latent),
                        label=f"surr {latent}",
                        color="red",
                        style="--",
                    ),
                ],
            )
            for latent in latent_vars
        ],
        PanelSpec(
            "orig gates",
            [
                TraceSpec(gate_data.sel(variable=name), label=name)
                for name in gate_data.coords["variable"].values
            ],
        ),
    ]


def plot_2d_attractor_comparison(
    orig_ds, surr_ds, comp_id, state_vars=["V", "latent1"]
):
    """
    オリジナルとサロゲートのアトラクタ（相平面）を重ねて描画し、ダイナミクスの一致度を可視化する。
    """
    fig = Figure()
    ax = fig.subplots()

    print(orig_ds["vars"].coords["variable"].values)
    print(orig_ds["vars"].sel(gate=True).coords["variable"].values)
    print(orig_ds["vars"].dims)
    print(orig_ds.coords)

    def extract_trajectory(ds):
        coords = []
        for var in state_vars:
            # Vはgate=False、それ以外（latent等）はgate=Trueから取得
            is_gate = var != "V"
            d = (
                ds["vars"]
                .sel(gate=is_gate, comp_id=comp_id, variable=var)
                .values.squeeze()
            )
            coords.append(d)
        return coords  # [x_array, y_array]

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

    # アスペクト比を自動調整（電位と潜在変数のスケールが違う場合が多いため 'equal' は避ける）
    fig.tight_layout()

    return fig


DrawFn = Callable[[xr.Dataset, xr.Dataset, xr.Dataset, int], Figure]
DRAW_MAP: dict = {
    "diff": lambda orig, surr, pre, comp_id: draw_engine(
        spec_diff(orig, pre, surr, surr_id=comp_id)
    ),
    "simple": lambda orig, surr, pre, comp_id: draw_engine(spec_simple(orig)),
    "attractor": lambda orig, surr, pre, comp_id: plot_2d_attractor_comparison(
        pre, surr, comp_id
    ),
}
