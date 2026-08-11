"""**描画宣言の型**。実体は `scripts/catalog.py` の `REPORT` に型のまま並ぶ
(設定ファイルは持たない = パースも未知キー検出も要らず、綴り間違いは import 時に
Python が落とす)。既定値と「どのキーがあるか」の単一源もここ。

図の**種類**は各ドメインの集約関数そのものから取る (関数名 = `kinds` のキー。
文字列を手で書き写さないので rename が自動で追従する)。marimo/mlflow 非依存。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..surrogate.figures import (
    closure_figs,
    neuron_graph_figs,
    preprocessor_figs,
    summary_df,
    train_figs,
)
from ..waveform import cell_figs, current_preview_fig
from .grid import compare_grid_fig, metric_fig, trace_grid_fig

# `ReportSpec.kinds` に書けるキー = 保存できる図/表の種類の単一源。
# **報告に載る種類の一覧はドメインを横断する** → 横断できる唯一の層 (report) が持つ。
# `cell_figs` だけは呼び出しに `wave_report` が付随する複合キーだが、キー名は
# `cell_figs` で足りる。
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

ALL_KINDS: tuple[str, ...] = tuple(f.__name__ for f in KIND_FUNCS)


@dataclass(frozen=True)
class DrawSpec:
    """1 系列の表示設定 (`ReportSpec.results` の 1 件)。**評価 (系列) ごとに固有の
    値** — 適用先が変われば comp 名も変わるのでグローバル既定を持たない
    (`eval_comp` 未指定はエラー図になる = 黙って何か描かず気付ける)。

    描画スタイルはここに無い (呼び出し側 `scripts/marimo.py` の定数の関心)。
    """

    eval_comp: str = ""  # 比較対象 comp (1 件)
    view_comps: tuple[str, ...] = ()  # 全 comp を並べる図の表示制限 (空=全部)
    detail_point: int = 0  # 詳細図 (diff/attractor/指標) を描く点の index
    spike_orig: int = 0
    spike_surr: int = 0
    # 点軸メトリクス図 (点が 2 つ以上のときだけ描く折れ線)
    metric: str = "spike_count"
    metric_yauto: bool = True
    metric_ymin: float = 0.0
    metric_ymax: float = 1.0

    @property
    def metric_ylim(self) -> tuple[float, float] | None:
        """折れ線の y レンジ (auto なら None = matplotlib 任せ)。"""
        return None if self.metric_yauto else (self.metric_ymin, self.metric_ymax)


@dataclass(frozen=True)
class CompareSpec:
    """既に回した複数系列を 1 枚の格子へ並べる宣言 (`ReportSpec.compares` の 1 件)。
    シミュを増やさず結果を参照するだけなので `eval_comp` を自分で持つ。"""

    evals: tuple[str, ...] = ()
    eval_comp: str = ""


@dataclass(frozen=True)
class ReportSpec:
    """描画宣言の全体。`results`/`compares` はどちらも系列名キーの dict。"""

    results: dict[str, DrawSpec] = field(default_factory=dict)
    compares: dict[str, CompareSpec] = field(default_factory=dict)
    kinds: dict[str, bool] = field(default_factory=dict)

    def wants(self, kind: str) -> bool:
        """この種類の図/表を保存するか。未指定キーは描く既定 (`kinds` に明示した
        ものだけが上書きされる)。"""
        return self.kinds.get(kind, True)

    def draw_for(self, name: str) -> DrawSpec:
        """系列名の表示設定 (未宣言なら既定の `DrawSpec`)。"""
        return self.results.get(name, DrawSpec())

    def targets(self, names: list[str]) -> list[tuple[str, DrawSpec]]:
        """描く対象 (系列名, 表示設定) 列。`results` が空なら手元の結果を全部描く
        既定、非空なら**そこに列挙した系列名だけ**へ絞り込む (artifact が増える
        ほど出力も比例して増えるのを避ける)。手元に無い名前は黙って落とす
        (参照は宣言、欠落は宣言とのズレであって結果の欠陥ではない)。"""
        return [
            (name, self.draw_for(name))
            for name in names
            if not self.results or name in self.results
        ]
