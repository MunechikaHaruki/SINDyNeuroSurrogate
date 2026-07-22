"""学習成果物 = 方程式の閉包項 (ansatz が空けた穴を埋める学習済みモデル)。

ansatz が方程式の骨格 (どの状態をどう並べ、どこを物理で解くか) を決め、その中で
データから決まる部分をこの型が持つ。実装差は「何で表現し、どう同定したか」だけ:

  `sindy.SINDyBundle` : ライブラリ項の疎な線形結合 (ξ)、疎回帰で同定
  (将来 UDE)          : NN、ODE 解を通した勾配降下で同定

契約は**閉包項の型を知らない側 (bundle) が呼ぶもの**だけ = `metrics()` 1 本。
ansatz は自分が同定した具体型を `Ansatz[C]` の型引数で知っているので、そこから
呼ぶもの (SINDy の ξ・feature 展開、閉包項の評価コスト) は契約に要らない。
共通して持っていても、型を知らない呼び出し側が要求しない限りここには載せない。
何を食わせて同定したか (学習入力の選択規則) も持たない — meta だけで決まるので
`Ansatz.train_source` から引ける (成果物に焼き付ける必要がない)。

surrogate 内 leaf (meta.py と同格)。
"""

from abc import ABC, abstractmethod


class Closure(ABC):
    @abstractmethod
    def metrics(self) -> dict[str, float]:
        """MLflow へ流すモデル指標 (表現ごとに中身は違う)。bundle が型を知らない
        まま呼ぶ唯一の窓口。"""
        ...
