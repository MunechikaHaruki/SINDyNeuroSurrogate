from ..dataset import Edge, NeuronGraph
from .compartments import COMPARTMENT_TEMPLATES

_hh = COMPARTMENT_TEMPLATES["hh"]
_passive = COMPARTMENT_TEMPLATES["passive"]
_traub = COMPARTMENT_TEMPLATES["traub"]

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
