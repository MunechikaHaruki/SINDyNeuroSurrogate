def define_models(definitions: dict):
    mc_models = {}
    target_nodes = {}

    for name, spec in definitions.items():
        # 1. ノード名からインデックスへのマップを動的に作成
        nodes_dict = spec["nodes"]
        node_names = list(nodes_dict.keys())
        name_to_idx = {n: i for i, n in enumerate(node_names)}

        # 2. MC_MODELS 形式の構築
        mc_models[name] = {
            "nodes": [nodes_dict[n] for n in node_names],
            "edges": [(name_to_idx[u], name_to_idx[v], g) for u, v, g in spec["edges"]],
            "stim_node": name_to_idx[spec["stim"]],
        }

        # 3. ターゲットノードのインデックス抽出
        target_nodes[name] = name_to_idx[spec["target"]]

    return mc_models, target_nodes


model_definitions = {
    "hh7": {
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
        "target": "h2",  # 最小限の「名前」による指定
    }
}
model_definitions = {
    "hh": {"nodes": {"soma": "hh"}, "edges": [], "stim": "soma", "target": "soma"},
    "php": {
        "nodes": {"p1": "passive", "h1": "hh", "p2": "passive"},
        "edges": [("p1", "h1", 1.0), ("h1", "p2", 0.7)],
        "stim": "p1",
        "target": "h1",
    },
    "hhp": {
        "nodes": {"h1": "hh", "h2": "hh", "p1": "passive"},
        "edges": [("h1", "h2", 1.0), ("h2", "p1", 0.7)],
        "stim": "h1",
        "target": "h2",
    },
    "phh": {
        "nodes": {"p1": "passive", "h1": "hh", "h2": "hh"},
        "edges": [("p1", "h1", 1.0), ("h1", "h2", 0.7)],
        "stim": "p1",
        "target": "h1",
    },
    "phhpp": {
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
        "target": "h2",
    },
    "pphhp": {
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
        "target": "h1",
    },
    "phhhp": {
        "nodes": {"p1": "passive", "h1": "hh", "h2": "hh", "h3": "hh", "p2": "passive"},
        "edges": [
            ("p1", "h1", 1.0),
            ("h1", "h2", 0.7),
            ("h2", "h3", 0.7),
            ("h3", "p2", 0.5),
        ],
        "stim": "p1",
        "target": "h2",
    },
    "hh7": {
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
        "target": "h2",
    },
}
# 実行
MC_MODELS, TARGET_NODES = define_models(model_definitions)
