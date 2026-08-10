"""**描画宣言 (`scripts/conf/draw.json`) を型へ落とす唯一の入口**。

`ReportSpec.from_dict` を通った後は dict も文字列キーも出てこない
(`DrawSpec`/`CompareSpec` として持ち回る) = 綴り間違いは読込時に落ち、以降は
型で守られる。既定値と「どのキーがあるか」の単一源もここ。

図の**種類**は `metrics.artifact.KIND_FUNCS` の関数名から取る (文字列を手で書き
写さないので rename が自動で追従する)。marimo/mlflow 非依存。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

from .artifact import KIND_FUNCS

# `draw.json` の `kinds` に書けるキー = 保存できる図/表の種類。
ALL_KINDS: tuple[str, ...] = tuple(f.__name__ for f in KIND_FUNCS)


def _build(cls: type, d: dict, where: str) -> Any:
    """dict → dataclass。未知のキーは黙って捨てず落とす (typo が「既定のまま
    描かれた」に化けるのを防ぐ)。"""
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(
            f"{where}: 未知のキー {sorted(unknown)} (使えるのは {sorted(known)})"
        )
    return cls(**d)


@dataclass(frozen=True)
class DrawSpec:
    """1 系列の表示設定 (`draw.json` の `results[]` 1 件)。**評価 (系列) ごとに固有の
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
    """既に回した複数系列を 1 枚の格子へ並べる宣言 (`draw.json` の `compare[]`)。
    シミュを増やさず結果を参照するだけなので `eval_comp` を自分で持つ。"""

    evals: tuple[str, ...] = ()
    eval_comp: str = ""


@dataclass(frozen=True)
class ReportSpec:
    """`draw.json` 全体。`results`/`compares` は名前キーの dict へ畳んである。"""

    results: dict[str, DrawSpec] = field(default_factory=dict)
    compares: dict[str, CompareSpec] = field(default_factory=dict)
    kinds: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """生 dict → 型。`results[]`/`compare[]` は配列 → 名前キーの dict
        (`eval`/`name` キー自体は要素から取り除く)。"""
        return cls(
            results={
                str(r["eval"]): _build(
                    DrawSpec,
                    {
                        k: tuple(v) if k == "view_comps" else v
                        for k, v in r.items()
                        if k != "eval"
                    },
                    f"results[{r['eval']}]",
                )
                for r in d.get("results", ())
            },
            compares={
                str(c["name"]): _build(
                    CompareSpec,
                    {
                        k: tuple(v) if k == "evals" else v
                        for k, v in c.items()
                        if k != "name"
                    },
                    f"compare[{c['name']}]",
                )
                for c in d.get("compare", ())
            },
            kinds={str(k): bool(v) for k, v in d.get("kinds", {}).items()},
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.from_dict(json.loads(path.read_text()))

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
