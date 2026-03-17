import numpy as np
import pysindy as ps

from neurosurrogate.modeling import neuron_core
from neurosurrogate.modeling.neuron_core import (
    FUNC_COST_MAP,
    HH_COST,
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
)


def make_gate_lib(funcs, is_product=False):
    """Gate単体、または Gate * y のペアを生成するファクトリ"""
    f_names = [f.__name__ for f in funcs]
    if not is_product:
        # 単体: lambda x: alpha_m(x)
        f_list = [f for f in funcs]
        n_list = [(lambda n: lambda x: f"{n}({x})")(n) for n in f_names]
    else:
        # 積: lambda x, y: alpha_m(x) * y
        f_list = [(lambda f: lambda x, y: f(x) * y)(f) for f in funcs]
        n_list = [(lambda n: lambda x, y: f"{n}({x})*{y}")(n) for n in f_names]
    return ps.CustomLibrary(library_functions=f_list, function_names=n_list)


def make_volt_lib(specs):
    """(累乗, 変数個数) のタプルリストから生成"""
    f_list, n_list = [], []

    # 1. 内部で「関数を作るための関数」を定義（pを固定するため）
    def create_u_p_v_w(p_val):
        return (
            lambda u, v, w: np.power(u, p_val) * v * w,
            lambda u, v, w: f"np.power({u}, {p_val}) * {v} * {w}",
        )

    def create_u_p_v(p_val):
        return (
            lambda u, v: np.power(u, p_val) * v,
            lambda u, v: f"np.power({u}, {p_val}) * {v}",
        )

    def create_u_p(p_val):
        return lambda u: np.power(u, p_val), lambda u: f"np.power({u}, {p_val})"

    # 2. ループで適切な関数を生成して追加
    for p, vars_count in specs:
        if vars_count == 2:
            f, n = create_u_p_v_w(p)
        elif vars_count == 1:
            f, n = create_u_p_v(p)
        else:
            f, n = create_u_p(p)

        f_list.append(f)
        n_list.append(n)

    return ps.CustomLibrary(library_functions=f_list, function_names=n_list)


INITIALIZED_SINDY = ps.SINDy(
    feature_library=ps.GeneralizedLibrary(
        [
            make_gate_lib([alpha_m, alpha_h, alpha_n], is_product=False),
            make_gate_lib(
                funcs=[alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n],
                is_product=True,
            ),
            make_volt_lib([(3, 2), (3, 1), (4, 1), (4, 0)]),  # 累乗, 追加変数の数
            ps.CustomLibrary(
                library_functions=[lambda x: x, lambda: 1],
                function_names=[lambda x: f"{x}", lambda: "1"],
            ),
        ],
        inputs_per_library=[  # [0,1,2]はV,g',u
            [0],
            [0, 1],
            [0, 1, 2],  # gate_product に V, m, h を渡す
            [0, 1, 2],  # base に V, m, h を渡す
        ],
    ),
    optimizer=ps.optimizers.STLSQ(threshold=0.01, normalize_columns=False, alpha=2.0),
)

COST_MAP = {
    "func": FUNC_COST_MAP,
    "orig": HH_COST,
}


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


# これが YAML や外部 config から渡されるイメージ
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

SINDY_MODEl = {"sindy": INITIALIZED_SINDY, "env": neuron_core, "target": TARGET_NODES}
