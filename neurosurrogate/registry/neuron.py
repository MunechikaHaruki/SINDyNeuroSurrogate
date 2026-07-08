from ..core.network import Edge, NeuronGraph
from .compartments import COMPARTMENT_TEMPLATES, HHParams, TraubParams
from .traub19 import build_traub19

_hh = COMPARTMENT_TEMPLATES["hh"]
_passive = COMPARTMENT_TEMPLATES["passive"]
_traub = COMPARTMENT_TEMPLATES["traub"]

# per-compartment パラメータ例
# soma: デフォルト (高 G_NA)、dendrite: G_NA/G_K 低減で発火閾値↑
_HH_DENDRITE_PARAMS = HHParams(G_NA=60.0, G_K=18.0, G_LEAK=0.5)

# Traub: soma (デフォルト) vs dendrite (Na/K_DR/K_A 低減)
_TRAUB_DENDRITE_PARAMS = TraubParams(g_Na=5.0, g_K_DR=10.0, g_K_A=1.0)

MCMODELS: dict[str, NeuronGraph] = {
    "hh": NeuronGraph(
        nodes=[_hh.with_name("soma")],
        edges=[],
        stim="soma",
    ),
    "traub": NeuronGraph(
        nodes=[_traub.with_name("soma")],
        edges=[],
        stim="soma",
    ),
    "php": NeuronGraph.chain(["passive", "hh", "passive"], [1.0, 0.7]),
    "hhp": NeuronGraph.chain(["hh", "hh", "passive"], [1.0, 0.7]),
    "pph": NeuronGraph.chain(["passive", "hh", "hh"], [1.0, 0.7]),
    "phhpp": NeuronGraph.chain(
        ["passive", "hh", "hh", "passive", "passive"], [1.0, 0.7, 0.7, 0.5]
    ),
    "pphhp": NeuronGraph.chain(
        ["passive", "passive", "hh", "hh", "passive"], [1.0, 0.7, 0.7, 0.5]
    ),
    "phhhp": NeuronGraph.chain(
        ["passive", "hh", "hh", "hh", "passive"], [1.0, 0.7, 0.7, 0.5]
    ),
    "hh_multi": NeuronGraph(
        nodes=[
            _hh.with_name("soma"),
            _hh.with_name("d1").with_params(_HH_DENDRITE_PARAMS),
            _hh.with_name("d2").with_params(_HH_DENDRITE_PARAMS),
        ],
        edges=[Edge("soma", "d1", 1.0), Edge("d1", "d2", 0.7)],
        stim="soma",
    ),
    "traub_multi": NeuronGraph(
        nodes=[
            _traub.with_name("soma"),
            _traub.with_name("d1").with_params(_TRAUB_DENDRITE_PARAMS),
            _traub.with_name("d2").with_params(_TRAUB_DENDRITE_PARAMS),
        ],
        edges=[Edge("soma", "d1", 1.0), Edge("d1", "d2", 0.7)],
        stim="soma",
    ),
    "traub19": build_traub19(),
    "hh7": NeuronGraph(
        nodes=[
            _passive.with_name("p1"),
            _hh.with_name("h1"),
            _hh.with_name("h2"),
            _hh.with_name("h3"),
            _hh.with_name("h4"),
            _passive.with_name("p2"),
            _passive.with_name("p3"),
        ],
        edges=[
            Edge("p1", "h1", 1.0),
            Edge("h1", "h2", 0.7),
            Edge("h2", "h3", 0.7),
            Edge("h2", "h4", 0.5),
            Edge("h3", "p2", 0.5),
            Edge("h4", "p3", 0.6),
        ],
        stim="p1",
    ),
}
