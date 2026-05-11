def build_model(neuron_spec: dict):

    nodes_dict = neuron_spec["nodes"]
    name_to_idx = {n: i for i, n in enumerate(nodes_dict.keys())}

    return {
        "name_to_idx_dict": name_to_idx,
        "nodes": list(nodes_dict.values()),
        "edges": [
            (name_to_idx[u], name_to_idx[v], g) for u, v, g in neuron_spec["edges"]
        ],
        "stim_node": name_to_idx[neuron_spec["stim"]],
    }


MCMODELS = {
    "hh": build_model({"nodes": {"soma": "hh"}, "edges": [], "stim": "soma"}),
    "php": build_model(
        {
            "nodes": {"p1": "passive", "h1": "hh", "p2": "passive"},
            "edges": [("p1", "h1", 1.0), ("h1", "p2", 0.7)],
            "stim": "p1",
        }
    ),
    "hhp": build_model(
        {
            "nodes": {"h1": "hh", "h2": "hh", "p1": "passive"},
            "edges": [("h1", "h2", 1.0), ("h2", "p1", 0.7)],
            "stim": "h1",
        }
    ),
    "pph": build_model(
        {
            "nodes": {"p1": "passive", "h1": "hh", "h2": "hh"},
            "edges": [("p1", "h1", 1.0), ("h1", "h2", 0.7)],
            "stim": "p1",
        }
    ),
    "phhpp": build_model(
        {
            "nodes": {
                "p1": "passive",
                "h1": "hh",
                "h2": "hh",
                "p2": "passive",
                "p3": "passive",
            },
            "edges": [
                ("p1", "h1", 1.0),
                ("h1", "h2", 0.7),
                ("h2", "p2", 0.7),
                ("p2", "p3", 0.5),
            ],
            "stim": "p1",
        }
    ),
    "pphhp": build_model(
        {
            "nodes": {
                "p1": "passive",
                "p2": "passive",
                "h1": "hh",
                "h2": "hh",
                "p3": "passive",
            },
            "edges": [
                ("p1", "p2", 1.0),
                ("p2", "h1", 0.7),
                ("h1", "h2", 0.7),
                ("h2", "p3", 0.5),
            ],
            "stim": "p1",
        }
    ),
    "phhhp": build_model(
        {
            "nodes": {
                "p1": "passive",
                "h1": "hh",
                "h2": "hh",
                "h3": "hh",
                "p2": "passive",
            },
            "edges": [
                ("p1", "h1", 1.0),
                ("h1", "h2", 0.7),
                ("h2", "h3", 0.7),
                ("h3", "p2", 0.5),
            ],
            "stim": "p1",
        }
    ),
    "hh7": build_model(
        {
            "nodes": {
                "p1": "passive",
                "h1": "hh",
                "h2": "hh",
                "h3": "hh",
                "h4": "hh",
                "p2": "passive",
                "p3": "passive",
            },
            "edges": [
                ("p1", "h1", 1.0),
                ("h1", "h2", 0.7),
                ("h2", "h3", 0.7),
                ("h2", "h4", 0.5),
                ("h3", "p2", 0.5),
                ("h4", "p3", 0.6),
            ],
            "stim": "p1",
        }
    ),
}
