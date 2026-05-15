# mypy: ignore-errors

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
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
) -> Figure:
    panels = [*spec[:-1], spec[-1].with_xlabel("Time [ms]")] if spec else spec

    n_rows = len(panels)
    fig = plt.figure()
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


_NODE_COLORS = {
    "hh": "#4C9BE8",
    "passive": "#A8D5A2",
}
_STIM_BORDER = "#E85C4C"
_NODE_SIZE = 4000
_NODE_LABEL_FONTSIZE = 30
_EDGE_WIDTH = 2.0
_EDGE_COLOR = "#666666"
_EDGE_LABEL_FONTSIZE = 25
_STIM_LINEWIDTH = 3.0


def view_neuron_graph(net, figsize=(8, 4)) -> Figure:
    """NeuronGraph をnetworkxで可視化する。ノード色はコンパートメント種別、赤枠はstimノード。"""
    G = nx.DiGraph()
    for c in net.nodes:
        G.add_node(c.name, type=c.type_name)
    for e in net.edges:
        G.add_edge(e.src, e.dst, weight=e.weight)
        G.add_edge(e.dst, e.src, weight=e.weight)

    pos = nx.spring_layout(G, seed=42)

    node_colors = [_NODE_COLORS.get(G.nodes[n]["type"], "#CCCCCC") for n in G.nodes]
    edge_labels = {(e.src, e.dst): f"{e.weight:.2g}" for e in net.edges}
    node_edge_colors = [_STIM_BORDER if n == net.stim else "white" for n in G.nodes]
    node_linewidths = [_STIM_LINEWIDTH if n == net.stim else 1.0 for n in G.nodes]

    fig = Figure(figsize=figsize)
    ax = fig.subplots()

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        linewidths=node_linewidths,
        node_size=_NODE_SIZE,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=_NODE_LABEL_FONTSIZE, font_weight="bold"
    )
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=[(e.src, e.dst) for e in net.edges],
        arrows=False,
        width=_EDGE_WIDTH,
        edge_color=_EDGE_COLOR,
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax, font_size=_EDGE_LABEL_FONTSIZE
    )

    legend_handles = [mpatches.Patch(color=c, label=t) for t, c in _NODE_COLORS.items()]
    legend_handles.append(
        mpatches.Patch(
            facecolor="white", edgecolor=_STIM_BORDER, linewidth=2, label="stim"
        )
    )
    ax.legend(handles=legend_handles, loc="best", frameon=True)
    ax.set_title("NeuronGraph")
    ax.axis("off")
    fig.tight_layout()
    return fig


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
