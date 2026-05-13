from .model_dataset import Edge, NeuronGraph, Node

MCMODELS: dict[str, NeuronGraph] = {
    "hh": NeuronGraph(
        nodes=[Node("soma", "hh")],
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
            Node("p1", "passive"),
            Node("h1", "hh"),
            Node("h2", "hh"),
            Node("h3", "hh"),
            Node("h4", "hh"),
            Node("p2", "passive"),
            Node("p3", "passive"),
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
