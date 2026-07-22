"""学習成果物 = 方程式の閉包項 (ansatz が空けた穴を埋める学習済みモデル)。

ansatz が方程式の骨格 (どの状態をどう並べ、どこを物理で解くか) を決め、その中で
データから決まる部分をこの型が持つ。実装差は「何で表現し、どう同定したか」だけ:

  `sindy.SINDyBundle` : ライブラリ項の疎な線形結合 (ξ)、疎回帰で同定
  (将来 UDE)          : NN、ODE 解を通した勾配降下で同定

契約は**閉包項の型を知らない側 (bundle / view) が呼ぶもの**だけ = `metrics()` と
`train_source` の 2 本。ansatz は自分が同定した具体型を `Ansatz[C]` の型引数で知って
いるので、そこから呼ぶもの (SINDy の ξ・feature 展開、閉包項の評価コスト) は契約に
要らない。共通して持っていても、型を知らない呼び出し側が要求しない限り載せない。

surrogate 内 leaf (meta.py と同格)。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ...core import access


@dataclass(frozen=True)
class TrainSource:
    """閉包項に何を食わせたかの仕様 = 学習データの再生成規則。

    軌道の実体は保存しない (`SurrogateMeta.dataset` から決定的に再現できる)。
    「どの comp の・先頭何本のゲートを取ったか」だけを閉包項が持てば、load 後でも
    meta と合わせて学習入力を組み直せる → marimo が run ロードのたびに学習データを
    再生成して描ける。列選択の知識は学習時に ansatz が決めてここへ畳み、ansatz の
    private に閉じ込めない。
    """

    comp_ids: list[int]  # 軌道を取った comp (comp_id 昇順)
    n_gate: int  # 先頭から何本のゲートを学習したか (残りは physics 側が解く)

    def gate(self, train_xr: xr.Dataset, comp_id: int) -> np.ndarray:
        """comp_id の学習ゲート行列 (time, n_gate)。"""
        return access.gate_matrix(train_xr, comp_id)[:, : self.n_gate]

    def stacked_gate(self, train_xr: xr.Dataset) -> np.ndarray:
        """全 comp を縦連結した学習ゲート行列 (preprocessor 学習用。preprocessor は
        時間微分を取らないので連結してよい — 閉包項の同定は軌道を分けて渡す)。"""
        return np.concatenate([self.gate(train_xr, i) for i in self.comp_ids], axis=0)


class Closure(ABC):
    # 学習入力の再生成仕様。同定時に ansatz が決めて載せる (実装は field で持つ)。
    train_source: TrainSource

    @abstractmethod
    def metrics(self) -> dict[str, float]:
        """MLflow へ流すモデル指標 (表現ごとに中身は違う)。bundle が型を知らない
        まま呼ぶ唯一の窓口。"""
        ...
