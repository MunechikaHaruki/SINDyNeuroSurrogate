import numpy as np


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

    result = {"exp": 0, "pow": 0, "dot": 0, "pm": 0, "divide": 0}

    cost_map = {
        "alpha": {
            "exp": 1,
            "+": 2,
        },
        "beta": {},
    }
    active_features = get_active_features(sindy_model)

    # まずは

    return result
