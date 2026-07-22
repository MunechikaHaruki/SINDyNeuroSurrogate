from __future__ import annotations

import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import sympy as sp
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure

from ..surrogate.closure.base import Closure
from ..surrogate.closure.sindy import SINDyBundle
from .engine import new_figure, place_legend

_NODE_COLORS = {
    "hh": "#4C9BE8",
    "passive": "#A8D5A2",
}
_SURR_COLOR = "#B57EDC"
_STIM_BORDER = "#E85C4C"
_NODE_SIZE = 4000
_NODE_LABEL_FONTSIZE = 30
_EDGE_WIDTH = 2.0
_EDGE_COLOR = "#666666"
_EDGE_LABEL_FONTSIZE = 25
_STIM_LINEWIDTH = 3.0
_COEF_DIGITS = 3  # 方程式表示の係数有効桁
_EQ_HEAD_TERMS = 3  # 見出しに出す先頭項数 (残りは \cdots)
_EQ_FONTSIZE = 8
_FEATURE_FONTSCALE = 1.3  # heatmap X軸 basis 関数 (TeX) をデフォルトの何倍にするか
_T = sp.Symbol("t")


def view_neuron_graph(net, surrogate_nodes=None, figsize=None) -> Figure:
    """NeuronGraph を networkx で可視化。ノード色=種別、赤枠=stim ノード。

    surrogate_nodes (置換対象ノード名集合) を渡すと該当ノードを紫で強調。
    """
    surrogate_nodes = surrogate_nodes or set()
    G = nx.DiGraph()
    for c in net.nodes:
        G.add_node(c.name, type=c.type.name)
    for e in net.edges:
        G.add_edge(e.src, e.dst, weight=e.weight)
        G.add_edge(e.dst, e.src, weight=e.weight)

    # BFS 深さを層に割当 → multipartite で層ごとに列。chain は直線、tree は階層列
    # (spring_layout の blob を回避)。根は stim でなく端点 (stim から最遠ノード) に
    # 取る → chain で stim が中央でも層が片方向に伸び 1 ノード/列の直線になる
    # (traub19: soma stim が中央でも c00..soma..c18 が一列)。
    from_stim = nx.single_source_shortest_path_length(G, net.stim)
    root = max(from_stim, key=lambda n: from_stim[n])
    depth = nx.single_source_shortest_path_length(G, root)
    for n in G.nodes:
        G.nodes[n]["layer"] = depth.get(n, 0)
    pos = nx.multipartite_layout(G, subset_key="layer")

    # figsize 未指定なら列数 (chain 長) / 最大同層ノード数からノードが重ならない寸法を
    # 自動算出。traub19 (19 列直線) を横長に伸ばして重なりを解消する。
    if figsize is None:
        layers = [G.nodes[n]["layer"] for n in G.nodes]
        n_cols = max(layers) + 1
        max_rows = max(layers.count(v) for v in set(layers))
        figsize = (max(8.0, n_cols * 1.3), max(4.0, max_rows * 1.8))

    node_colors = [
        _SURR_COLOR
        if n in surrogate_nodes
        else _NODE_COLORS.get(G.nodes[n]["type"], "#CCCCCC")
        for n in G.nodes
    ]
    edge_labels = {(e.src, e.dst): f"{e.weight:.2g}" for e in net.edges}
    node_edge_colors = [_STIM_BORDER if n == net.stim else "white" for n in G.nodes]
    node_linewidths = [_STIM_LINEWIDTH if n == net.stim else 1.0 for n in G.nodes]

    fig = new_figure(figsize=figsize)
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
    if surrogate_nodes:
        legend_handles.append(mpatches.Patch(color=_SURR_COLOR, label="surrogate"))
    legend_handles.append(
        mpatches.Patch(
            facecolor="white", edgecolor=_STIM_BORDER, linewidth=2, label="stim"
        )
    )
    place_legend(ax, legend_handles)
    ax.set_title("NeuronGraph")
    ax.axis("off")
    return fig


def _latex(e: sp.Basic) -> str:
    """sympy 式 → latex 本体。レート名 alpha_m_hh の下付き "m hh" は sympy が空白で
    繋ぐが、mathtext は下付き内の空白を詰めて α_mhh と読めなくなる → 表示時のみ
    model を括弧に整形する (α_{m(hh)})。"""
    return re.sub(r"_\{(\w+) (\w+)\}", r"_{\1(\2)}", sp.latex(e))


def tex(e: sp.Basic) -> str:
    """sympy 式 → インライン数式 (matplotlib mathtext / marimo の md 共通記法)。"""
    return f"${_latex(e)}$"


def closure_figs(closure: Closure) -> list[tuple[str, Figure]]:
    """閉包項の中身図 (識別子付き)。中身の描き方は表現ごとに違い、共通の図は無い
    (SINDy=ξ heatmap、NN 表現なら重み分布など) → 型で振り分ける。図を持たない表現
    は空列を返し、呼び出し側は保存/表示に流すだけで済む。"""
    if isinstance(closure, SINDyBundle):
        return [("model", view_model(closure))]
    return []


def view_model(result: SINDyBundle, figsize=(15, 3)):
    xi_matrix = np.asarray(result.xi)
    fig = new_figure(figsize=figsize)
    ax = fig.subplots()

    vmin = np.min(xi_matrix)
    vmax = np.max(xi_matrix)

    if vmin == 0 and vmax == 0:
        vmin, vmax = -1.0, 1.0

    # linthresh=1.0: -1.0〜1.0 の微小係数は線形スケール
    norm = SymLogNorm(linthresh=1.0, vmin=vmin, vmax=vmax, base=10)

    sns.heatmap(
        xi_matrix,
        cmap="coolwarm",
        center=0,
        norm=norm,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        annot=False,
    )

    # 図題は suptitle に上げ、その下 (heatmap との間) に各 target の式を抜粋表示。
    fig.suptitle("SINDy Coefficients (SymLog Scale)")
    ax.set_title("\n".join(equation_texs(result)), fontsize=_EQ_FONTSIZE)

    if len(result.targets) == xi_matrix.shape[0]:
        ax.set_yticks(np.arange(len(result.targets)) + 0.5)
        ax.set_yticklabels([tex(s) for s in result.targets], rotation=0)
        ax.set_ylabel("Target Variables")

    if len(result.feature_exprs) == xi_matrix.shape[1]:
        ax.set_xticks(np.arange(xi_matrix.shape[1]) + 0.5)
        ax.set_xticklabels(
            [tex(e) for e in result.feature_exprs],
            rotation=45,
            ha="right",
            fontsize=plt.rcParams["font.size"] * _FEATURE_FONTSCALE,
        )
        ax.set_xlabel("Library Features")

    return fig


def equation_texs(bundle: SINDyBundle) -> list[str]:
    """target ごとの d(target)/dt = Σ ξ·θ を先頭数項だけ切り出した数式 (図の見出し
    用の抜粋。全項は heatmap 本体が示す)。係数の丸めと 0 係数落としは表示都合で、
    xi 本体は触らない。"""
    exprs = bundle.feature_exprs
    texs = []
    for target, row in zip(bundle.targets, bundle.xi, strict=True):
        terms = [
            sp.Float(c, _COEF_DIGITS) * expr
            for c, expr in zip(row, exprs, strict=True)
            if c != 0
        ]
        head = sp.Eq(
            sp.Derivative(target, _T),
            sum(terms[:_EQ_HEAD_TERMS], sp.S.Zero),
            evaluate=False,
        )
        tail = r" + \cdots" if len(terms) > _EQ_HEAD_TERMS else ""
        texs.append(f"${_latex(head)}{tail}$")
    return texs
