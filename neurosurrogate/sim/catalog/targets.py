"""`SimSpec.target` が引く適用先モデルのカタログ。組み立ては `neurons/generate.py`。"""

from ...core.network import Compartment, Edge, NeuronGraph
from ...neurons.compartments.hh import HH_TYPE, PASSIVE_TYPE, HHParams
from ...neurons.compartments.traub import TRAUB_TYPE, TraubParams
from ...neurons.generate import build_traub19, chain
from ...neurons.traub19 import DEND_STIM_IDX

# per-compartment パラメータ例
# soma: デフォルト (高 G_NA)、dendrite: G_NA/G_K 低減で発火閾値↑
_HH_DENDRITE_PARAMS = HHParams(G_NA=60.0, G_K=18.0, G_LEAK=0.5)

# Traub: soma (デフォルト) vs dendrite (Na/K_DR/K_A 低減)
_TRAUB_DENDRITE_PARAMS = TraubParams(g_Na=5.0, g_K_DR=10.0, g_K_A=1.0)

MCMODELS: dict[str, NeuronGraph] = {
    "hh": NeuronGraph(
        nodes=[Compartment(name="soma", type=HH_TYPE)],
        edges=[],
        stim="soma",
    ),
    "traub": NeuronGraph(
        nodes=[Compartment(name="soma", type=TRAUB_TYPE)],
        edges=[],
        stim="soma",
    ),
    "php": chain(["passive", "hh", "passive"], [1.0, 0.7]),
    "hhp": chain(["hh", "hh", "passive"], [1.0, 0.7]),
    "pph": chain(["passive", "hh", "hh"], [1.0, 0.7]),
    "phhpp": chain(["passive", "hh", "hh", "passive", "passive"], [1.0, 0.7, 0.7, 0.5]),
    "pphhp": chain(["passive", "passive", "hh", "hh", "passive"], [1.0, 0.7, 0.7, 0.5]),
    "phhhp": chain(["passive", "hh", "hh", "hh", "passive"], [1.0, 0.7, 0.7, 0.5]),
    "hh_multi": NeuronGraph(
        nodes=[
            Compartment(name="soma", type=HH_TYPE),
            Compartment(name="d1", type=HH_TYPE, params=_HH_DENDRITE_PARAMS),
            Compartment(name="d2", type=HH_TYPE, params=_HH_DENDRITE_PARAMS),
        ],
        edges=[Edge("soma", "d1", 1.0), Edge("d1", "d2", 0.7)],
        stim="soma",
    ),
    "traub_multi": NeuronGraph(
        nodes=[
            Compartment(name="soma", type=TRAUB_TYPE),
            Compartment(name="d1", type=TRAUB_TYPE, params=_TRAUB_DENDRITE_PARAMS),
            Compartment(name="d2", type=TRAUB_TYPE, params=_TRAUB_DENDRITE_PARAMS),
        ],
        edges=[Edge("soma", "d1", 1.0), Edge("d1", "d2", 0.7)],
        stim="soma",
    ),
    # 全 comp 同一 traub 型 → 全ノード置換対象 (元の traub.c と同じ soma 注入)。
    "traub19": build_traub19(),
    # soma だけ traub 型に残し dendrite をダミー型 traub_ に。comp_type=traub の学習を
    # そのまま適用すると soma 1 ノードだけ置換される (適用先で範囲を絞る→preset 不変)。
    "traub19_soma": build_traub19(soma_only=True),
    # 同上 (soma だけ置換対象) だが電流注入を dendrite に。soma 非注入で dend → soma
    # 伝播を surrogate soma が再現できるかを見る。
    "traub19_soma_dendstim": build_traub19(soma_only=True, stim_idx=DEND_STIM_IDX),
    "hh7": NeuronGraph(
        nodes=[
            Compartment(name="p1", type=PASSIVE_TYPE),
            Compartment(name="soma", type=HH_TYPE),
            Compartment(name="h2", type=HH_TYPE),
            Compartment(name="h3", type=HH_TYPE),
            Compartment(name="h4", type=HH_TYPE),
            Compartment(name="p2", type=PASSIVE_TYPE),
            Compartment(name="p3", type=PASSIVE_TYPE),
        ],
        edges=[
            Edge("p1", "soma", 1.0),
            Edge("soma", "h2", 0.7),
            Edge("h2", "h3", 0.7),
            Edge("h2", "h4", 0.5),
            Edge("h3", "p2", 0.5),
            Edge("h4", "p3", 0.6),
        ],
        stim="p1",
    ),
}
