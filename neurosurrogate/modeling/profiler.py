import re

import numpy as np

cost_map = {
    "alpha_m": {
        "exp": 1,
        "div": 1,
        "pm": 2,  # 2.5 - 0.1v (1回分) と 分母の - 1.0
        "mul": 2,  # 0.1 * v (1回分)
    },
    "beta_m": {
        "exp": 1,
        "div": 1,
        "pm": 1,  # -v
        "mul": 1,  # 4.0 * exp
    },
    "alpha_h": {
        "exp": 1,
        "div": 1,
        "pm": 1,
        "mul": 1,
    },
    "beta_h": {
        "exp": 1,
        "div": 1,
        "pm": 2,  # 3.0 - 0.1v と + 1.0
        "mul": 1,  # 0.1 * v
    },
    "alpha_n": {
        "exp": 1,
        "div": 1,
        "pm": 2,
        "mul": 2,
    },
    "beta_n": {
        "exp": 1,
        "div": 1,
        "pm": 1,
        "mul": 1,
    },
    "a_n": {"exp": 1, "div": 1, "pm": 2, "mul": 2},
}


def get_active_features(sindy_model):
    # 係数が非ゼロのインデックスを取得
    coefs = sindy_model.coefficients()
    # 各変数（v, m, h等）の微分方程式において、一つでも非ゼロ係数を持つ項のマスク
    active_mask = np.any(coefs != 0, axis=0)

    # すべての特徴量名を取得し、アクティブなものだけに絞り込む
    all_features = sindy_model.get_feature_names()
    active_features = [f for i, f in enumerate(all_features) if active_mask[i]]
    return active_features


def static_calc_cost(sindy_model):
    """ "
    expの演算回数が~~~,+の演算回数が~~~みたいに計算
    """

    surrogate_raw = {
        "exp": 0,
        "div": 0,
        "pm": 0,
        "mul": 0,
    }
    coef = sindy_model.coefficients()
    nnz = np.count_nonzero(coef)
    surrogate_raw["mul"] = nnz
    surrogate_raw["pm"] = max(0, nnz - coef.shape[0])

    active_features = get_active_features(sindy_model)

    for feature in active_features:
        work_str = feature

        # ステップ1: 物理関数を特定して消去 (Pop)
        for func_name, costs in cost_map.items():
            if func_name in work_str:
                # 物理関数が見つかった回数分コストを加算
                count = work_str.count(func_name)
                for op, val in costs.items():
                    surrogate_raw[op] += val * count
                # 解析済みとして文字列から消す
                work_str = work_str.replace(func_name, "")

        # ステップ2: np.power を特定して消去
        # 正規表現でべき乗数を抽出し、(n-1)回の乗算として加算
        powers = re.findall(r"np\.power\(.*?, (\d+)\)", work_str)
        for p in powers:
            surrogate_raw["mul"] += int(p) - 1
        work_str = re.sub(r"np\.power\(.*?, \d+\)", "", work_str)

        # ステップ3: 残った文字列（残渣）から演算子をカウント
        # ここに残っている '*' や '+' は、項同士の結合に使われているものだけ
        surrogate_raw["mul"] += work_str.count("*")
        surrogate_raw["pm"] += work_str.count("+") + work_str.count("-")
        # u や V_soma などの変数名は無視される

    # 2. オリジナルモデルのコスト取得
    original_raw = get_original_hh_cost()

    # 3. 階層化辞書の構築
    result = {}

    # surrogate / original カテゴリ
    for k, v in surrogate_raw.items():
        result[f"cost/surrogate/{k}"] = v
    for k, v in original_raw.items():
        result[f"cost/original/{k}"] = v

    # diff / reduction_pct カテゴリ
    # 基本演算(exp, div, pm, mul)について比較
    for k in original_raw.keys():
        v_orig = original_raw[k]
        v_surr = surrogate_raw.get(k, 0)

        # 差分 (削減量)
        result[f"cost/diff/{k}"] = v_surr - v_orig

    return result


def get_original_hh_cost():
    """
    提供された calc_deriv_hh / hh3 のコードを静的にトレースした演算コスト。
    """
    res = {"exp": 0, "div": 0, "pm": 0, "mul": 0}

    # 1. alpha/beta (6個分)
    for func in ["alpha_m", "beta_m", "alpha_h", "beta_h", "alpha_n", "beta_n"]:
        for op, val in cost_map[func].items():
            res[op] += val

    # 2. Gating variables (m0, h0, n0, tau_m, tau_h, tau_n) の計算
    # m0 = alpha / (alpha + beta) -> 1pm, 1div
    # tau = 1 / (alpha + beta) -> 1pm, 1div
    res["pm"] += (1 + 1) * 3
    res["div"] += (1 + 1) * 3

    # 3. calc_deriv_hh 内部
    res["pm"] += 1  # v_rel = v - p.E_REST
    res["pm"] += 1
    res["mul"] += 1  # i_leak
    res["pm"] += 1
    res["mul"] += 5  # i_na (m*m*m*h*(v-E))
    res["pm"] += 1
    res["mul"] += 5  # i_k (n*n*n*n*(v-E))

    res["pm"] += 4
    res["div"] += 1  # dvar[0]
    res["pm"] += 2 * 3
    res["mul"] += 1 * 3
    res["div"] += 1 * 3  # dvar[1-3]

    # # 4. calc_deriv_hh3 (コンパートメント接続部)
    # res["pm"] += 4  # I_pre, I_post の計算 (各2pm)
    # res["pm"] += 4
    # res["mul"] += 4
    # res["div"] += 2  # v_pre, v_post の微分

    return res
