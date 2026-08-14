"""**描画への入力**: 1 レポートをどう描くか (`Tuning`)。

ドメイン (`neurosurrogate.report`) には置かない — 描画関数はどれも必要な値を素の
引数で受け取り、この束を知らない。値を与えるのは `scripts/marimo.py` の widget
1 箇所で、束を解いて描画層へ渡すのは `scripts/mlflow_io` の各 module = **UI の
つまみ 1 組**という入口側の関心にそろえる。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tuning:
    """**1 レポートの描画条件**。既定値と「どのキーがあるか」の単一源はここで、値を
    与える場所は `scripts/marimo.py` の widget 1 箇所だけ (カタログは「何を回すか」
    しか持たない = 描き方は図を見て決め直すものなので寿命が違う)。

    先頭の `eval_comp` だけ既定値が無い — 適用先が変われば comp 名も変わるので、
    系列を選んだ後に決まる。残りは既定のままでもレポートが出る。

    **何を描くかは宣言しない**: モデル側の図はその run が自分について描けるもの
    (`surrogate.figures.surrogate_figs` が bundle の型から解く)、評価側の図は結果の形
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
