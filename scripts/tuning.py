"""**描画への入力**: 1 レポートをどう描くか (`Tuning`)。

ドメインの成果物生成 module には置かない — 描画関数はどれも必要な値を素の
引数で受け取り、この束を知らない。値を与えるのは `scripts/marimo.py` の widget
1 箇所で、束を解いて描画層へ渡すのは `scripts/mlflow_io` の各 module = **UI の
つまみ 1 組**という入口側の関心にそろえる。

**つまみの選択肢 (`comp_names`) と widget からの組立 (`Tuning.from_widgets`) もここ**
= キーの綴りと既定値と選択肢が 1 file に閉じる (marimo のセルは widget を置いて
`.value` を渡すだけ)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from catalog import SERIES


def comp_names(series_name: str | None) -> list[str]:
    """系列名 → その系列の適用先に在る comp 名 (未選択は空)。comp のつまみ
    (`eval_comp` / `view_comps`) の選択肢はこれ = 適用先と噛み合わない comp を
    選べない。名前の解決は適用先を知る `SimSpec.net` に任せる。"""
    return sorted(SERIES[series_name].spec.net.names) if series_name else []


@dataclass(frozen=True)
class Tuning:
    """**1 レポートの描画条件**。既定値と「どのキーがあるか」の単一源はここで、値を
    与える場所は `scripts/marimo.py` の widget 1 箇所だけ (カタログは「何を回すか」
    しか持たない = 描き方は図を見て決め直すものなので寿命が違う)。

    先頭の `eval_comp` だけ既定値が無い — 適用先が変われば comp 名も変わるので、
    系列を選んだ後に決まる。残りは既定のままでもレポートが出る。

    **何を描くかは宣言しない**: モデル側の図はその run が自分について描けるもの
    (`artifact.bundle.surrogate_artifacts` が bundle の型から解く)、評価側の図は結果の形
    (点が 2 つ以上なら折れ線が出る) で決まる。図の種類名がこの型に出てこないのが
    不変条件。
    """

    eval_comp: str  # 比較対象 comp (1 件)
    view_comps: tuple[str, ...] = ()  # 全 comp を並べる図の表示制限 (空=全部)
    metric: str = "spike_count"  # 点軸の折れ線に使う指標
    detail_point: int = 0  # 詳細図 (diff/attractor/指標) を描く点の index
    spike_orig: int = 0  # 特徴量比較に使う原系の何本目のスパイクか
    spike_surr: int = 0  # 同じく置換系
    metric_ylim: tuple[float, float] | None = None  # 折れ線の y レンジ (None=auto)

    @classmethod
    def from_widgets(cls, values: dict[str, Any]) -> Tuning:
        """**widget の値 1 組 → 描き方 1 値**。y レンジの 3 つ (auto/下限/上限) は
        `metric_ylim: tuple | None` へ畳み、comp 未選択 (系列未選択) は空文字のまま
        渡して描画側で設定誤りとして落とす。UI の都合をこの型に持ち込まないための
        変換なので、キーの綴りを知る側 (= 既定値を持つ側) に置く。"""
        return cls(
            eval_comp=values["eval_comp"] or "",
            view_comps=tuple(values["view_comps"]),
            metric=values["metric"],
            detail_point=int(values["detail_point"]),
            spike_orig=int(values["spike_orig"]),
            spike_surr=int(values["spike_surr"]),
            metric_ylim=None if values["yauto"] else (values["ymin"], values["ymax"]),
        )
