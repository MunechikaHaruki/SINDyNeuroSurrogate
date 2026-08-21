"""同定器へ渡す直前の入力一式。

`Ansatz` が組み `Closure` の同定入口が受ける**受け渡しの型**であって契約ではない
(抽象メソッドを 1 つも持たない) ので、契約を集めた `parts/__init__.py` でなく
ここに単独で置く。両側がここを見る = 契約と実装の間に横たわる 3 つ目のもの。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class TrainInputs:
    """列順 = ansatz が組んだ列構造。fit が作って流し、view が同じものを描く。
    軌道は comp ごとに分けたまま持つ (縦連結は偽微分)。時間軸と出所 comp は持たない
    (training_data / spec.train_comp_ids が源)。

    x_names/u_names : 列の表示名 (状態列 / 入力列)。
    """

    x_names: list[str]
    u_names: list[str]
    x: list[np.ndarray]  # 各 (time, len(x_names))、comp_ids 順
    u: list[np.ndarray]  # 各 (time, len(u_names))

    def target_symbols(self) -> list[sp.Symbol]:
        """状態列の記号 (列名がそのまま記号)。"""
        return [sp.Symbol(v) for v in self.x_names]

    def input_symbols(self) -> list[sp.Symbol]:
        """入力列の記号。"""
        return [sp.Symbol(v) for v in self.u_names]
