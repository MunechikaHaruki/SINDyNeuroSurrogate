"""初期状態がノード自身の params で解かれることの回帰テスト。

CompartmentType が初期状態を型レベルの定数で持っていた頃、traub19 の全 comp が
soma (default params) の Ca 濃度 XI を初期値に取り、traub.c 参照から最大 84 mV
乖離していた。初期状態は params からの導出量であり、型に焼いてはいけない。
"""

import numpy as np
import pytest

from neurosurrogate.compartments.traub import (
    TRAUB_STATE_NAMES,
    TraubParams,
    _traub_inf_v,
)
from neurosurrogate.models import MCMODELS
from neurosurrogate.models.traub19 import SOMA_IDX

XI_IDX = 1 + TRAUB_STATE_NAMES.index("XI")  # init は [V, *gates]
Q_IDX = 1 + TRAUB_STATE_NAMES.index("Q")
TRAUB19_NODES = MCMODELS["traub19"].nodes


def test_v_init_is_node_leak_potential():
    for node in TRAUB19_NODES:
        assert node.init[0] == node.resolved_params.V_LEAK


def test_xi_init_matches_traub_c_formula():
    """traub.c initialize(): XI = -phi[i]*area[i]*g_Ca[i]*S²R*(V-V_Ca)/Beta。"""
    for node in TRAUB19_NODES:
        p: TraubParams = node.resolved_params
        s = float(_traub_inf_v("S", p.V_LEAK))
        r = float(_traub_inf_v("R", p.V_LEAK))
        assert node.init[XI_IDX] == pytest.approx(
            -p.phi_area * p.g_Ca * s * s * r * (p.V_LEAK - p.V_Ca) / p.Beta
        )


def test_ca_free_compartments_start_with_zero_calcium():
    """g_Ca=0 の comp に Ca は湧かない。型レベル定数では踏めた不変条件。"""
    zero_ca = [n for n in TRAUB19_NODES if n.resolved_params.g_Ca == 0.0]
    assert zero_ca, "g_Ca=0 の comp が traub19 に存在するはず"
    for node in zero_ca:
        assert node.init[XI_IDX] == 0.0
        assert node.init[Q_IDX] == 0.0


def test_xi_init_varies_across_compartments():
    """全 comp が soma 値を共有していた退行を直接検出する。"""
    xis = np.array([n.init[XI_IDX] for n in TRAUB19_NODES])
    assert len(np.unique(xis)) > 1
    assert not np.allclose(xis, TRAUB19_NODES[SOMA_IDX].init[XI_IDX])
