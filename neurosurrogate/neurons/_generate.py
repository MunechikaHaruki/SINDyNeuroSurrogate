"""NeuronGraph の生成関数。組んだ結果のカタログは `__init__.py`。"""

from ..core.network import Compartment, Edge, NeuronGraph
from .compartments.traub import TRAUB_TYPE
from .traub19 import NC, g_axial, name_at, params_at


def build_traub19() -> NeuronGraph:
    """19-comp Traub モデルを組む (per-comp 定数は `traub19.py`)。

    全 comp 同一 traub 型。注入ノードは形態でなく `SimSpec.stim` が、置換範囲は
    `EvalSeries.replace_targets` が決める = **形態はどちらも知らない**。
    """
    nodes = [
        Compartment(name=name_at(i), type=TRAUB_TYPE, params=params_at(i))
        for i in range(NC)
    ]
    edges = [Edge(name_at(i), name_at(i + 1), g_axial(i)) for i in range(NC - 1)]
    return NeuronGraph(nodes=nodes, edges=edges)
