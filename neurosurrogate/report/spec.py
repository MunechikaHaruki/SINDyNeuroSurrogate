"""**描画宣言の型**。実体は `scripts/catalog.py` の `REPORT` に型のまま並ぶ
(設定ファイルは持たない = パースも未知キー検出も要らず、綴り間違いは import 時に
Python が落とす)。既定値と「どのキーがあるか」の単一源もここ。

**1 レポート = 1 系列 × N モデル**: ある系列の電流たちで複数の surrogate を比べる、
という 1 つの問いが 1 レポート。系列を跨いで並べる図は持たない (複数系列を 1 枚に
畳むと「どの電流で比べたか」が図の中で二重になる)。

**何を描くかは宣言しない**: モデル側の図はその run が自分について描けるもの
(`surrogate.figures.surrogate_figs` が bundle の型から解く)、評価側の図は結果の形
(点が 2 つ以上なら折れ線が出る) で決まる。図の種類名がこの層に出てこないのが不変条件。
marimo/mlflow 非依存。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Report:
    """1 系列 (= 1 レポート) の描画条件。**研究条件そのもの**で、カタログに書いて
    差分に残すもの (どの comp を比較の主役に据えるか、どの指標で追うか)。

    `eval_comp` に既定値は無い — 適用先が変われば comp 名も変わるので、系列ごとに
    書くしかない。
    """

    eval_comp: str  # 比較対象 comp (1 件)
    view_comps: tuple[str, ...] = ()  # 全 comp を並べる図の表示制限 (空=全部)
    metric: str = "spike_count"  # 点軸の折れ線に使う指標


@dataclass(frozen=True)
class Tuning:
    """図を見ながら回す**つまみ** (marimo の widget 状態)。`Report` と分けてあるのは
    寿命が違うから — こちらは描画のたびに変わってよく、カタログには残らない。

    既定値だけで描ける = つまみを触らなければ `Report` だけでレポートが出る。
    """

    detail_point: int = 0  # 詳細図 (diff/attractor/指標) を描く点の index
    spike_orig: int = 0  # 特徴量比較に使う原系の何本目のスパイクか
    spike_surr: int = 0  # 同じく置換系
    metric_ylim: tuple[float, float] | None = None  # 折れ線の y レンジ (None=auto)


# つまみを触らない描画の既定 (frozen なので共有して問題ない)。
NO_TUNING = Tuning()
