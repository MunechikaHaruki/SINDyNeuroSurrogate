from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr


def build_indices(net: dict, compartments: dict):
    """
    ネットワーク配線図 (MC_MODELS) と 部品カタログ (COMPARTMENT_TEMPLATES) から
    Numba用のインデックス配列と初期値配列を自動生成する。
    """
    nodes = net["nodes"]
    N = len(nodes)

    gate_offsets = np.full(N, -1, dtype=np.int32)
    init_list = []

    coord_comp_id = []
    coord_variable = []
    coord_gate = []

    # --- 修正1 & 2: 通常のリストとして初期化 ---
    ids_list = {}
    for k in compartments.keys():
        ids_list[k] = []

    # [Pass 1] 全ノードのVの初期値を配置し、IDを振り分ける
    for i, node_type in enumerate(nodes):
        # 電位の初期値を追加
        init_list.append(compartments[node_type]["init"][0])
        ids_list[node_type].append(i)  # 普通のリストなのでappend可能

        coord_comp_id.append(i)
        coord_variable.append(compartments[node_type]["vars"][0])  # 例: "V"
        coord_gate.append(compartments[node_type]["gate"][0])  # 例: False

    # [Pass 2] ゲート変数のオフセット計算と初期値の配置
    current_offset = N
    for i, node_type in enumerate(nodes):
        gate_inits = compartments[node_type]["init"][1:]

        # ゲートの変数名とフラグを取得
        gate_vars = compartments[node_type]["vars"][1:]
        gate_flags = compartments[node_type]["gate"][1:]
        # ゲート変数が存在する場合のみオフセットを記録
        if len(gate_inits) > 0:
            gate_offsets[i] = current_offset
            init_list.extend(gate_inits)
            current_offset += len(gate_inits)

            # ゲートのラベル情報を追加
            coord_comp_id.extend(
                [i] * len(gate_inits)
            )  # 例: [0, 0, 0] (m, h, nが全て同じcomp_id)
            coord_variable.extend(gate_vars)  # 例: ["M", "H", "N"] または ["latent1"]
            coord_gate.extend(gate_flags)  # 例: [True, True, True]

    # 最後に、集めたIDリストを一気にNumPy配列(int32)に変換する
    ids = defaultdict(lambda: np.array([], dtype=np.int32))
    for k, v in ids_list.items():
        ids[k] = np.array(v, dtype=np.int32)

    return {
        "ids": ids,
        "gate_offsets": gate_offsets,
        "init": np.array(init_list, dtype=np.float64),
        "coords": {
            "comp_id": coord_comp_id,
            "variable": coord_variable,
            "gate": coord_gate,
        },
    }


def set_coords(raw, u, coords, dt):
    mindex = pd.MultiIndex.from_arrays(
        [
            coords["comp_id"],
            coords["variable"],
            coords["gate"],
        ],
        names=("comp_id", "variable", "gate"),
    )
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex, "features")

    # 2. Dataset 作成時に一括で定義する
    dataset = xr.Dataset(
        {
            "vars": (("time", "features"), raw),
            "I_ext": (("time"), u),
        },
        coords={
            "time": np.arange(len(u)) * dt,
            **mindex_coords,  # ここで一気にマルチインデックス化
        },
    )
    return dataset


def set_i_internal(dataset, C_matrix, I_ext_2d):
    # コンパートメント間を流れる電流の系間を流れる電流の計算
    v_dataset = dataset["vars"].sel(gate=False).sortby("comp_id")
    V_data = v_dataset.values  # 形状: (time, N)
    I_internal_np = V_data @ C_matrix
    # コンパートメントに対し、直接入力される電流をたす
    I_internal_np = I_internal_np + I_ext_2d
    # xarray に格納
    dataset["I_internal"] = xr.DataArray(
        I_internal_np,  # (N, time) の形状にするため転置
        coords={
            "time": dataset.time,
            "node_id": np.arange(I_internal_np.shape[1]),
        },
        dims=["time", "node_id"],
    )
