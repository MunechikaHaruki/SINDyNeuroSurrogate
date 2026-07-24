"""学習成果物 = 方程式の閉包項 (ansatz が空けた穴を埋める学習済みモデル)。

ansatz が方程式の骨格 (どの状態をどう並べ、どこを物理で解くか) を決め、その中で
データから決まる部分をこの型が持つ。実装差は「何で表現し、どう同定したか」だけ:

  `sindy.SINDyBundle` : ライブラリ項の疎な線形結合 (ξ)、疎回帰で同定
  (将来 UDE)          : NN、ODE 解を通した勾配降下で同定

契約 = **全実装が持ち、呼び手が具体型に依らず一律に問うもの** = `metrics()` と
`opcost()`。前者は bundle が、後者は ansatz が呼ぶが、どちらも「引数なしで問う」
形で表現に依らない (中身の構成は違っても問い方は同一)。逆に SINDy の ξ・feature
展開のような**具体型固有の引き出し**は `Ansatz[C]` が型を知って行う → 契約に載せ
ない。何を食わせて同定したか (学習入力の選択規則) も持たない — meta だけで決まる
ので `Ansatz.train_source` から引ける (成果物に焼き付ける必要がない)。

surrogate 内 leaf (meta.py と同格)。
"""

from abc import ABC, abstractmethod

from ...core.opcost import OpCost


class Closure(ABC):
    @abstractmethod
    def metrics(self) -> dict[str, float]:
        """MLflow へ流すモデル指標 (表現ごとに中身は違う)。bundle が型を知らない
        まま呼ぶ窓口。"""
        ...

    @abstractmethod
    def opcost(self) -> OpCost:
        """閉包項 1 回の評価コスト。構成は表現ごとに違うが問い方は引数なしで同一 →
        全実装が持ち ansatz が一律に呼ぶので契約に載せる。"""
        ...
