"""描画の宣言 (`draw.json`) の型。表示設定 + 並べ方のスキーマを知る唯一の場所。

`eval.spec.parse_evals` が計算入力 (`eval.json`) に対してやるのと同じ役目 — **dict
を見るのはここまで**で、以降 (`report.model_report`/`report.eval_report`) は型
(`DrawSpec`/`ReportSpec`/`CompareSpec`) で渡す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from ..core.network import NeuronGraph
from ..eval.run import SimKey
from ..eval.store import SimResult
from . import select
from .artifact import KIND_FUNCS


@dataclass(frozen=True, kw_only=True)
class DrawSpec:
    """表示設定のスキーマを知る唯一の場所 (UI は同名キーの dict を組むだけ = キーの
    意味と既定値の源はここ 1 つ)。**評価 (系列名) ごとに固有の値** (適用先が変われば
    comp 名も変わる) — グローバルな既定値を持たない (描画スタイルは呼び出し側
    `scripts/marimo.py` の定数の関心で、ここには持たない)。"""

    eval_comp: str = ""  # 比較対象 comp (1 件)。適用先ごとに違うので既定を持たない
    view_comps: tuple[str, ...] = ()  # 全 comp を並べる図の表示制限 (空=全部)
    detail_point: int = 0  # 詳細図 (diff/attractor/指標) を描く点の index
    spike_orig: int = 0
    spike_surr: int = 0
    # 点軸メトリクス図 (点が 2 つ以上のときだけ描く折れ線)
    metric: str = "spike_count"
    metric_yauto: bool = True
    metric_ymin: float = 0.0
    metric_ymax: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> DrawSpec:
        """`draw.json` の `results[]` 1 件 → 型 (欠落キーは既定値)。
        **dict を見るのはここまで**で、以降は型で渡す。"""
        return cls(
            eval_comp=str(d.get("eval_comp") or ""),
            view_comps=tuple(str(c) for c in d.get("view_comps", ())),
            detail_point=int(d.get("detail_point", cls.detail_point)),
            spike_orig=int(d.get("spike_orig", cls.spike_orig)),
            spike_surr=int(d.get("spike_surr", cls.spike_surr)),
            metric=str(d.get("metric", cls.metric)),
            metric_yauto=bool(d.get("metric_yauto", cls.metric_yauto)),
            metric_ymin=float(d.get("metric_ymin", cls.metric_ymin)),
            metric_ymax=float(d.get("metric_ymax", cls.metric_ymax)),
        )

    def view_comp_ids(self, net: NeuronGraph) -> list[int] | None:
        """全 comp を並べる図に描く comp。UI では名前、描画側は comp_id で受ける。
        空選択 = 制限なし (None)。"""
        return [net.name_to_idx(c) for c in self.view_comps] or None

    def metric_ylim(self) -> tuple[float, float] | None:
        """点軸メトリクス図の y レンジ (auto なら None = matplotlib 任せ)。"""
        return None if self.metric_yauto else (self.metric_ymin, self.metric_ymax)


@dataclass(frozen=True, kw_only=True)
class CompareSpec:
    """複数の評価結果を 1 枚の格子へ縦に並べる図の宣言 (行=評価、列=点)。

    シミュ仕様ではなく**既に回した結果への参照**なので描画側の型 (compare を足しても
    回るシミュは eval entry を書いた分だけ)。`evals` は系列名 (`SimSpec.name`) 列。
    `eval_comp` は比較する compare 自身の設定 (系列ごとの `results[]` には
    紐付かないグローバル設定を持たないので、compare 自身が持つ)。名前は
    `ReportSpec.compares` の dict key が持つのでここには持たない。
    """

    evals: list[str]
    eval_comp: str

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            evals=[str(s) for s in d["evals"]],
            eval_comp=str(d["eval_comp"]),
        )


# `results`/`kinds` を指定しなければ手元の結果を全種描く既定 = 何も書かなくても
# 動く (最初の 1 回はここから)。絞り込みたくなったら明示的に列挙する。
# キー名は `KIND_FUNCS` (`metrics/artifact/__init__.py`) の関数名から取る = 文字列
# を手で書き写さないので rename しても自動で追従する。
ALL_KINDS = tuple(f.__name__ for f in KIND_FUNCS)


@dataclass(frozen=True, kw_only=True)
class ReportSpec:
    """描画宣言全体 (`draw.json`) の型 = 系列名ごとの設定 + compare + 保存する種類の
    絞り込み。**dict を見るのはここまで**で、以降 (`report.model_report`/
    `report.eval_report`) は型で渡す (`eval.spec.parse_evals` が計算入力に対して
    やるのと同じ役目)。
    """

    results: dict[str, DrawSpec] = field(default_factory=dict)
    compares: dict[str, CompareSpec] = field(default_factory=dict)
    kinds: dict[str, bool] = field(
        default_factory=lambda: dict.fromkeys(ALL_KINDS, True)
    )

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        kinds = (
            {str(k): bool(v) for k, v in d["kinds"].items()}
            if "kinds" in d
            else dict.fromkeys(ALL_KINDS, True)
        )
        return cls(
            results={
                str(r["eval"]): DrawSpec.from_dict(r) for r in d.get("results", ())
            },
            compares={
                str(c["name"]): CompareSpec.from_dict(c) for c in d.get("compare", ())
            },
            kinds=kinds,
        )

    def draw_for(self, name: str) -> DrawSpec:
        """系列名の表示設定 (`results[]` に無ければ `DrawSpec` の型既定値。
        `eval_comp` が空だと `_eval_report_one` がエラー図を出す = 未指定は
        黙って何か描くのでなく気付ける形にする)。"""
        return self.results.get(name, DrawSpec())

    def for_results(
        self, results: dict[SimKey, SimResult]
    ) -> list[tuple[str, DrawSpec]]:
        """描く対象 (系列名, draw) 列。`results` が空なら手元の結果を全部描く既定。
        非空なら **`results` に列挙した系列名だけへ絞り込む** (artifact が増えるほど
        draw 出力も比例して増えるのを避ける)。手元に無い名前は黙って落とす
        (`_compare_report` の「参照は宣言、欠落は宣言とのズレ」という扱いと揃える)。
        """
        names = select.series(results)
        if not self.results:
            return [(name, self.draw_for(name)) for name in names]
        return [(name, self.draw_for(name)) for name in names if name in self.results]

    def wants(self, kind: str) -> bool:
        """この種類の図/表を保存するか (`kinds` による絞り込み)。キーに無いものは
        false 扱い (`kinds` を明示指定したら列挙しなかった種類は描かない)。"""
        return self.kinds.get(kind, False)
