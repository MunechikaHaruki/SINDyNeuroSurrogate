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


def calc_dynamic_metrics(orig_ds, surr_ds, comp_id, dt):
    orig_v = orig_ds["vars"].sel(gate=False, comp_id=comp_id).to_numpy().squeeze()
    surr_v = surr_ds["vars"].sel(gate=False, comp_id=comp_id).to_numpy().squeeze()

    orig_peaks, _ = find_peaks(orig_v, height=0.0)
    surr_peaks, _ = find_peaks(surr_v, height=0.0)

    return {
        **_calc_waveform_metrics(orig_v, surr_v),
        **_calc_spike_metrics(orig_peaks, surr_peaks, dt),
        **_calc_windowless_spike_metrics(orig_v, surr_v, dt),
    }


def _calc_waveform_metrics(orig_v, surr_v):
    return {
        "rmse": float(np.sqrt(np.mean((orig_v - surr_v) ** 2))),
        "mae": float(np.mean(np.abs(orig_v - surr_v))),
    }


def _calc_spike_metrics(orig_peaks, surr_peaks, dt):
    orig_isi = np.diff(orig_peaks) * dt if len(orig_peaks) >= 2 else None
    surr_isi = np.diff(surr_peaks) * dt if len(surr_peaks) >= 2 else None

    return {
        "orig_spike_count": len(orig_peaks),
        "surr_spike_count": len(surr_peaks),
        "spike_count_diff": abs(len(orig_peaks) - len(surr_peaks)),
        "latency_error": _latency_error(orig_peaks, surr_peaks, dt),
        "orig_mean_isi": float(np.mean(orig_isi)) if orig_isi is not None else np.nan,
        "orig_std_isi": float(np.std(orig_isi)) if orig_isi is not None else np.nan,
        "surr_mean_isi": float(np.mean(surr_isi)) if surr_isi is not None else np.nan,
        "surr_std_isi": float(np.std(surr_isi)) if surr_isi is not None else np.nan,
        "periodicity_gap": abs(np.mean(orig_isi) - np.mean(surr_isi))
        if orig_isi is not None and surr_isi is not None
        else np.nan,
    }


def _latency_error(orig_peaks, surr_peaks, dt):
    if len(orig_peaks) > 0 and len(surr_peaks) > 0:
        return float(abs(orig_peaks[0] - surr_peaks[0]) * dt)
    return np.nan


def _calc_windowless_spike_metrics(orig_v, surr_v, dt):
    """
    波形全体から物理的特徴量の分布（中央値）を抽出して比較する。
    窓関数の設定を排除し、ダイナミクス自体の再現性をロバストに評価する。
    """
    # 1. 微分波形の事前計算
    orig_dvdt = np.diff(orig_v) / dt
    surr_dvdt = np.diff(surr_v) / dt

    # 2. 特徴量抽出の共通ヘルパー
    def get_median_peak(signal, height):
        peaks, _ = find_peaks(signal, height=height)
        return np.median(signal[peaks]) if len(peaks) > 0 else np.nan

    # 3. 各物理指標の中央値を一括算出
    # (対象信号, 信号名, 検出閾値) のリストで定義
    feature_configs = [
        (orig_v, surr_v, "amp", 0.0),  # 電位ピーク (mV)
        (orig_dvdt, surr_dvdt, "max_dvdt", 10.0),  # 最大立上り速度 (mV/ms)
        (
            -orig_dvdt,
            -surr_dvdt,
            "min_dvdt",
            10.0,
        ),  # 最大立下り速度 (mV/ms) ※反転して検出
    ]

    results = {}
    for o_sig, s_sig, key, thresh in feature_configs:
        o_med = get_median_peak(o_sig, thresh)
        s_med = get_median_peak(s_sig, thresh)

        # 片方でもピークがなければ NaN 誤差、あれば絶対誤差
        # ※ min_dvdt の場合、get_median_peak は正の値を返すが、
        # 誤差計算において絶対値をとるため、符号反転の考慮は不要
        results[f"median_{key}_error"] = float(abs(o_med - s_med))

    return results
