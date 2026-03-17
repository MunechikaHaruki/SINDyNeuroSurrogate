import re
from collections import Counter

import numpy as np
from scipy.signal import find_peaks


def get_active_features(sindy_model):
    # 係数が非ゼロのインデックスを取得
    coefs = sindy_model.coefficients()
    # 各変数（v, m, h等）の微分方程式において、一つでも非ゼロ係数を持つ項のマスク
    active_mask = np.any(coefs != 0, axis=0)

    # すべての特徴量名を取得し、アクティブなものだけに絞り込む
    all_features = sindy_model.get_feature_names()
    active_features = [f for i, f in enumerate(all_features) if active_mask[i]]
    return active_features


def static_calc_cost(sindy_model, cost_map, original_cost):
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
    nnz = np.count_nonzero(coef).item()
    surrogate_raw["mul"] = nnz
    surrogate_raw["pm"] = max(0, nnz - int(coef.shape[0]))

    active_features = get_active_features(sindy_model)

    for feature in active_features:
        # 騒々しく失敗する (Fail Fast, Fail Loudly)
        if feature not in cost_map:
            raise ValueError(
                f"未知の基底関数 '{feature}' が見つかりました。"
                "注入された cost_map にこの基底関数の定義を追加してください。"
            )

        # 辞書から直接コストを引いて足すだけ（超高速＆確実）
        feature_cost = cost_map[feature]
        for op, val in feature_cost.items():
            surrogate_raw[op] += val

    # 3. 階層化辞書の構築
    result = {}

    # surrogate / original カテゴリ
    for k, v in surrogate_raw.items():
        result[f"cost/surrogate/{k}"] = v
    for k, v in original_cost.items():
        result[f"cost/original/{k}"] = v

    # diff / reduction_pct カテゴリ
    # 基本演算(exp, div, pm, mul)について比較
    for k in original_cost.keys():
        v_orig = original_cost[k]
        v_surr = surrogate_raw.get(k, 0)

        # 差分 (削減量)
        result[f"cost/diff/{k}"] = v_surr - v_orig

    return result


def build_feature_cost_map(feature_names: list, base_cost_map: dict) -> dict:
    """
    SINDyが生成したすべての基底関数名(文字列)を解析し、
    O(1)で引ける完全一致のコスト辞書を自動生成する。
    """
    feature_cost_map = {}

    # 部分一致バグを防ぐため、名前が長い順にソートして処理
    sorted_funcs = sorted(base_cost_map.keys(), key=len, reverse=True)

    for feature in feature_names:
        cost = Counter({"exp": 0, "div": 0, "pm": 0, "mul": 0})
        work_str = feature

        # ステップ1: 基礎物理関数のコストを抽出し、文字列から消去
        for func_name in sorted_funcs:
            count = work_str.count(func_name)
            if count > 0:
                for op, val in base_cost_map[func_name].items():
                    cost[op] += val * count
                work_str = work_str.replace(func_name, "")

        # ステップ2: べき乗 (np.power) の展開コスト
        # np.power(x, 3) なら (3-1)=2回の掛け算
        powers = re.findall(r"np\.power\(.*?, (\d+)\)", work_str)
        cost["mul"] += sum(int(p) - 1 for p in powers)
        work_str = re.sub(r"np\.power\(.*?, \d+\)", "", work_str)

        # ステップ3: 残った演算子 (* や +) のカウント
        cost["mul"] += work_str.count("*")
        cost["pm"] += work_str.count("+") + work_str.count("-")

        # 辞書に登録
        feature_cost_map[feature] = dict(cost)

    return feature_cost_map


def calc_dynamic_metrics(orig_v, surr_v, dt):
    """
    オリジナル波形とサロゲート波形の力学系・神経科学的メトリクスを計算する。
    """
    metrics = {}

    # 1. 基本的な波形誤差
    metrics["rmse"] = np.sqrt(np.mean((orig_v - surr_v) ** 2))
    metrics["mae"] = np.mean(np.abs(orig_v - surr_v))

    # 2. スパイク検出 (一般的な活動電位を想定し、0mVを閾値とする)
    # ※ モデルの静止膜電位やピークスケールに合わせて height は調整してください
    orig_peaks, _ = find_peaks(orig_v, height=0.0)
    surr_peaks, _ = find_peaks(surr_v, height=0.0)

    # スパイク数の比較
    metrics["orig_spike_count"] = len(orig_peaks)
    metrics["surr_spike_count"] = len(surr_peaks)
    metrics["spike_count_diff"] = abs(len(orig_peaks) - len(surr_peaks))

    # 3. 発火潜時 (最初のスパイクまでの時間) の誤差
    if len(orig_peaks) > 0 and len(surr_peaks) > 0:
        orig_latency = orig_peaks[0] * dt
        surr_latency = surr_peaks[0] * dt
        metrics["latency_error"] = abs(orig_latency - surr_latency)
    else:
        metrics["latency_error"] = np.nan  # どちらかが発火しなかった場合

    # 4. ISI (Inter-Spike Interval: スパイク間隔) の計算
    # スパイクが2回以上ないとISIは計算できないため分岐
    if len(orig_peaks) >= 2:
        orig_isi = np.diff(orig_peaks) * dt
        metrics["orig_mean_isi"] = np.mean(orig_isi)
        metrics["orig_std_isi"] = np.std(orig_isi)
    else:
        metrics["orig_mean_isi"] = np.nan
        metrics["orig_std_isi"] = np.nan

    if len(surr_peaks) >= 2:
        surr_isi = np.diff(surr_peaks) * dt
        metrics["surr_mean_isi"] = np.mean(surr_isi)
        metrics["surr_std_isi"] = np.std(surr_isi)
    else:
        metrics["surr_mean_isi"] = np.nan
        metrics["surr_std_isi"] = np.nan

    # ISIの誤差 (両方に十分なスパイクがある場合)
    if len(orig_peaks) >= 2 and len(surr_peaks) >= 2:
        metrics["mean_isi_error"] = abs(
            metrics["orig_mean_isi"] - metrics["surr_mean_isi"]
        )
    else:
        metrics["mean_isi_error"] = np.nan

    return metrics
