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


def build_feature_library():
    library = ps.GeneralizedLibrary(
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
    )
    env = neuron_core
    cost_map = {
        "func": FUNC_COST_MAP,
        "orig": HH_COST,
    }
    return library, env, cost_map
