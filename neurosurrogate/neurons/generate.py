"""NeuronGraph の生成関数。組んだ結果のカタログは `sim/catalog/targets.py`。"""

from collections import Counter

from ..core.network import Compartment, Edge, NeuronGraph
from .compartments import COMPARTMENT_TYPES
from .compartments.traub import TRAUB_DUMMY_TYPE, TRAUB_TYPE
from .traub19 import NC, SOMA_IDX, area_at, g_axial, name_at, params_at


def chain(
    node_types: list[str],
    weights: list[float],
    stim: int = 0,
) -> NeuronGraph:
    """type 名リストから直鎖 NeuronGraph 構築。

    ノード名は型の頭文字 + 0始まり連番。ただし最初の非 passive (=細胞体) は "soma"
    と命名し全モデル共通の soma 規約に揃える (train_comp_identifier 既定と一致)。
    例: ["passive","hh","passive"] → ["p0","soma","p1"]
    """
    expected = len(node_types) - 1
    assert len(weights) == expected, (
        f"weights の長さは len(node_types) - 1 = {expected} 必要"
    )
    soma_idx = next(
        (i for i, t in enumerate(node_types) if t != "passive"),
        None,
    )
    assert soma_idx is not None, "chain に非 passive (soma) ノードが必要"
    counters: Counter = Counter()
    nodes = []
    for i, t in enumerate(node_types):
        if i == soma_idx:
            name = "soma"
        else:
            name = f"{t[0]}{counters[t[0]]}"
            counters[t[0]] += 1
        nodes.append(Compartment(name=name, type=COMPARTMENT_TYPES[t]))
    return NeuronGraph(
        nodes=nodes,
        edges=[
            Edge(nodes[i].name, nodes[i + 1].name, w) for i, w in enumerate(weights)
        ],
        stim=nodes[stim].name,
    )


def build_traub19(
    soma_only: bool = False,
    stim_idx: int = SOMA_IDX,
) -> NeuronGraph:
    """19-comp Traub モデルを組む (per-comp 定数は `traub19.py`、変種は `sim/catalog`)。

    soma_only=False (既定) は全 comp 同一 traub 型。True にすると soma だけ traub 型に
    残り dendrite はダミー型 (別型 = 置換対象外) になる → comp_type=traub の学習が soma
    1 ノードだけへ適用される。stim_idx で注入ノードを変える (area scale も注入ノードの
    ものに合わせる)。
    """
    nodes = [
        Compartment(
            name=name_at(i),
            type=TRAUB_DUMMY_TYPE if soma_only and i != SOMA_IDX else TRAUB_TYPE,
            params=params_at(i),
        )
        for i in range(NC)
    ]
    edges = [Edge(name_at(i), name_at(i + 1), g_axial(i)) for i in range(NC - 1)]
    # 外部電流を密度 [μA/cm^2] → 絶対 [μA] に変換 (kernel 内で /area して密度に戻す)
    return NeuronGraph(
        nodes=nodes,
        edges=edges,
        stim=name_at(stim_idx),
        stim_area_scale=area_at(stim_idx),
    )
