"""Traub 1991 CA3 pyramidal cell (19 compartments)。

C reference: tmp/dataset_utils/traub/traub.c と代数的等価。

- 各 compartment に per-comp params (g_*, phi*area, area)
- edge weight = 隣接軸 conductance g_axial [μS] (対称量) → graph_laplacian symmetric
- kernel 側で u_t/area で密度化 → C の /area[i] を吸収
- stim: soma (index=8) に絶対電流 [μA] 注入
"""

import math

from ..core.network import Compartment, Edge, NeuronGraph
from .compartments.traub import TRAUB_DUMMY_TYPE, TRAUB_TYPE, TraubParams

# --- traub.c の per-compartment 定数 (19要素) ---

_G_NA = [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 15.0, 30.0, 15.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # noqa: E501  # fmt: skip
_G_K_DR = [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 5.0, 15.0, 5.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # noqa: E501  # fmt: skip
_G_K_A = [0.0] * 8 + [5.0] + [0.0] * 10  # index 8 のみ 5.0
_G_K_C = [0.0, 5.0, 5.0, 10.0, 10.0, 10.0, 5.0, 20.0, 10.0, 20.0, 5.0, 15.0, 15.0, 15.0, 15.0, 15.0, 5.0, 5.0, 0.0]  # noqa: E501  # fmt: skip
_G_K_AHP = [0.0] + [0.8] * 17 + [0.0]
_G_CA = [0.0, 5.0, 5.0, 12.0, 12.0, 12.0, 5.0, 8.0, 4.0, 8.0, 5.0, 17.0, 17.0, 17.0, 10.0, 10.0, 5.0, 5.0, 0.0]  # noqa: E501  # fmt: skip
_G_LEAK = 0.1
_PHI = [7769.0] * 7 + [34530.0, 17402.0, 26404.0] + [5941.0] * 9
_RAD = [2.89e-4] * 8 + [4.23e-4] + [2.42e-4] * 10
_LEN = [1.20e-2] * 8 + [1.25e-2] + [1.10e-2] * 10
_AREA = [2.188e-5] * 8 + [3.320e-5] + [1.673e-5] * 10
_RI = 0.1  # KΩ·cm

NC = 19
SOMA_IDX = 8
_SOMA_NAME = "soma"  # 細胞体は全モデル共通で "soma" と命名 (SOMA_IDX の comp)


def _name_at(i: int) -> str:
    """細胞体 (SOMA_IDX) は "soma"、他は c00..c18。"""
    return _SOMA_NAME if i == SOMA_IDX else f"c{i:02d}"


def _params_at(i: int) -> TraubParams:
    return TraubParams(
        g_leak=_G_LEAK,
        g_Na=_G_NA[i],
        g_Ca=_G_CA[i],
        g_K_DR=_G_K_DR[i],
        g_K_A=_G_K_A[i],
        g_K_AHP=_G_K_AHP[i],
        g_K_C=_G_K_C[i],
        phi_area=_PHI[i] * _AREA[i],
        area=_AREA[i],
    )


def _g_axial(i: int) -> float:
    """comp i と i+1 の間の軸方向 conductance [μS] (対称量)。"""
    r_i = _RI * _LEN[i] / (math.pi * _RAD[i] ** 2)
    r_ip1 = _RI * _LEN[i + 1] / (math.pi * _RAD[i + 1] ** 2)
    return 2.0 / (r_i + r_ip1)


# dend 刺激版の注入先。soma (8) 隣の active な apical proximal dendrite で、
# soma へ確実に電流が伝播する (g_Na=15)。
_DEND_STIM_IDX = 9


def _build_traub19(
    soma_only: bool = False,
    stim_idx: int = SOMA_IDX,
) -> NeuronGraph:
    """19-comp Traub モデルを組む内部ヘルパ (公開するのは下の TRAUB19_MODELS)。

    soma_only=False (既定) は全 comp 同一 traub 型。True にすると soma だけ traub 型に
    残り dendrite はダミー型 (別型 = 置換対象外) になる → comp_type=traub の学習が soma
    1 ノードだけへ適用される。stim_idx で注入ノードを変える (area scale も注入ノードの
    ものに合わせる)。変種の意味は TRAUB19_MODELS 側。
    """
    nodes = [
        Compartment(
            name=_name_at(i),
            type=TRAUB_DUMMY_TYPE if soma_only and i != SOMA_IDX else TRAUB_TYPE,
            params=_params_at(i),
        )
        for i in range(NC)
    ]
    edges = [Edge(_name_at(i), _name_at(i + 1), _g_axial(i)) for i in range(NC - 1)]
    # 外部電流を密度 [μA/cm^2] → 絶対 [μA] に変換 (kernel 内で /area して密度に戻す)
    return NeuronGraph(
        nodes=nodes,
        edges=edges,
        stim=_name_at(stim_idx),
        stim_area_scale=_AREA[stim_idx],
    )


# 19-comp Traub の 3 変種を宣言的に定義 (__init__ は MCMODELS へ merge するだけ)。
# 変種差 = (dendrite を置換対象にするか, 電流注入先) の 2 軸だが、意図は名前で示す。
TRAUB19_MODELS: dict[str, NeuronGraph] = {
    # 全 comp 同一 traub 型 → 全ノード置換対象 (元の traub.c と同じ soma 注入)。
    "traub19": _build_traub19(),
    # soma だけ traub 型に残し dendrite をダミー型 traub_ に。comp_type=traub の学習を
    # そのまま適用すると soma 1 ノードだけ置換される (適用先で範囲を絞る→preset 不変)。
    "traub19_soma": _build_traub19(soma_only=True),
    # 同上 (soma だけ置換対象) だが電流注入を dendrite に。soma 非注入で dend → soma
    # 伝播を surrogate soma が再現できるかを見る。
    "traub19_soma_dendstim": _build_traub19(soma_only=True, stim_idx=_DEND_STIM_IDX),
}
