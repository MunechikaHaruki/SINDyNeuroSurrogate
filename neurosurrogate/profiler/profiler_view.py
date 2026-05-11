# mypy: ignore-errors

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure

# ── データ構造 ────────────────────────────────────────────────


@dataclass
class TraceSpec:
    data: xr.DataArray
    label: str | None = None
    color: str | None = None
    style: str = "-"

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.time.values, self.data.values.squeeze()


@dataclass
class PanelSpec:
    ylabel: str
    traces: list[TraceSpec] | TraceSpec = field(default_factory=list)
    xlabel: str | None = None

    def __post_init__(self):
        if isinstance(self.traces, TraceSpec):
            self.traces = [self.traces]

    def has_legend(self) -> bool:
        return any(tr.label for tr in self.traces)

    def with_xlabel(self, label: str) -> PanelSpec:
        return PanelSpec(ylabel=self.ylabel, traces=self.traces, xlabel=label)


# ── レンダラ ──────────────────────────────────────────────────
def draw_engine(
    spec: list[PanelSpec],
    figsize_width: int = 10,
) -> Figure:
    panels = [*spec[:-1], spec[-1].with_xlabel("Time [ms]")] if spec else spec

    n_rows = len(panels)
    fig = Figure(figsize=(figsize_width, 2 * n_rows))
    axs = fig.subplots(nrows=n_rows, ncols=1, sharex=True)
    if n_rows == 1:
        axs = [axs]

    for ax, p in zip(axs, panels):
        for tr in p.traces:
            x, y = tr.xy()
            ax.plot(x, y, label=tr.label, color=tr.color, linestyle=tr.style)
        ax.set_ylabel(p.ylabel)
        if p.has_legend():
            ax.legend()
        if p.xlabel:
            ax.set_xlabel(p.xlabel)

    fig.tight_layout()
    return fig


# ── Spec ビルダ ───────────────────────────────────────────────


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


def view_model(xi_matrix, feature_names=None, target_names=None, figsize=(15, 3)):
    xi_matrix = np.asarray(xi_matrix)
    fig, ax = plt.subplots(figsize=figsize)

    # 係数の最大・最小値からカラーマップの範囲を決定
    vmin = np.min(xi_matrix)
    vmax = np.max(xi_matrix)

    # 全て0の場合はエラーを避けるためのフォールバック
    if vmin == 0 and vmax == 0:
        vmin, vmax = -1.0, 1.0

    # 対称対数スケールの設定
    # linthresh=1.0 により、-1.0から1.0の微小な係数は線形スケールとして扱う
    norm = SymLogNorm(linthresh=1.0, vmin=vmin, vmax=vmax, base=10)

    # center=0 に設定することで、係数が0（スパース化により消去された項）が白になる
    sns.heatmap(
        xi_matrix,
        cmap="coolwarm",
        center=0,
        norm=norm,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        annot=False,  # 桁数が大きいため、文字入れはデフォルトでオフ
    )

    ax.set_title("SINDy Coefficients (SymLog Scale)")

    # Y軸のラベル設定（予測ターゲット）
    if target_names is not None and len(target_names) == xi_matrix.shape[0]:
        ax.set_yticks(np.arange(len(target_names)) + 0.5)
        ax.set_yticklabels(target_names, rotation=0)
        ax.set_ylabel("Target Variables")

    # X軸のラベル設定（ライブラリの項）
    if feature_names is not None and len(feature_names) == xi_matrix.shape[1]:
        ax.set_xticks(np.arange(len(feature_names)) + 0.5)
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Library Features")

    fig.tight_layout()
    return fig


def plot_2d_attractor_comparison(
    orig_ds, surr_ds, comp_id, state_vars=["V", "latent1"], figsize=(8, 7)
):
    """
    オリジナルとサロゲートのアトラクタ（相平面）を重ねて描画し、ダイナミクスの一致度を可視化する。
    """
    fig, ax = plt.subplots(figsize=figsize)

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
