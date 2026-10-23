"""区画の式とシミュレータの約束: 初期値は区画の params から解き、電流は密度で受ける。"""

from dataclasses import replace as dc_replace

import numpy as np
import pytest

from neurosurrogate.physics.cells import (
    NETS,
    TRAUB,
    TRAUB19_SOMA,
    TRAUB_STATES,
    Compartment,
    traub19_params,
)
from neurosurrogate.physics.simulate import (
    SimSpec,
    _areas,
    _laplacian,
    potential_matrix,
    simulate,
)

TRAUB19 = NETS["traub19"].nodes
XI = 1 + TRAUB_STATES.index("XI")  # init は [V, *states]
Q = 1 + TRAUB_STATES.index("Q")
_SPEC = SimSpec(
    target="traub19",
    current_type="lin&steady",
    dt=0.01,
    current_params={"silence_duration": 1.0, "duration": 5.0, "value": 5.0},
)


def test_initial_state_is_solved_from_each_compartments_params() -> None:
    """V は自身の V_LEAK、XI は traub.c の式。g_Ca=0 の区画に Ca は湧かない。"""
    for i, node in enumerate(TRAUB19):
        p = traub19_params(i)
        assert node.init[0] == p.V_LEAK
        s, r = (
            node.init[1 + TRAUB_STATES.index("S")],
            node.init[1 + TRAUB_STATES.index("R")],
        )
        expected = -p.phi_area * p.g_Ca * s * s * r * (p.V_LEAK - p.V_Ca) / p.Beta
        assert node.init[XI] == pytest.approx(expected)
        if p.g_Ca == 0.0:
            assert node.init[XI] == node.init[Q] == 0.0
    xis = np.array([n.init[XI] for n in TRAUB19])
    assert not np.allclose(xis, TRAUB19[TRAUB19_SOMA].init[XI])


def test_dv_does_not_depend_on_area() -> None:
    """kernel は電流密度を受けるので、面積が違っても dV/dt は同じ。"""
    soma = TRAUB19[TRAUB19_SOMA]
    x = np.array(soma.init[1:])
    p = traub19_params(TRAUB19_SOMA)
    wide = p._replace(area=p.area * 10)
    dv, _ = TRAUB.kernel(p, 2.5, soma.init[0], x)
    dv_wide, _ = TRAUB.kernel(wide, 2.5, soma.init[0], x)
    assert float(dv) == float(dv_wide)


def test_replaced_compartments_keep_their_area() -> None:
    net = NETS["traub19"]
    other = dc_replace(TRAUB, name="other")
    swapped = dc_replace(
        net, nodes=[Compartment(c.name, other, c.params) for c in net.nodes]
    )
    assert np.allclose(_areas(swapped), _areas(net))


def test_i_internal_divides_the_coupling_by_the_receiving_area() -> None:
    net = NETS["traub19"]
    laplacian = _laplacian(net)
    assert np.allclose(laplacian, laplacian.T)
    assert np.allclose(_areas(net), [traub19_params(i).area for i in range(19)])
    ds = simulate(_SPEC.run_input())
    expected = potential_matrix(ds) @ laplacian / _areas(net)
    expected[:, net.index(_SPEC.stim)] += _SPEC.current()
    assert np.allclose(ds["I_internal"].to_numpy(), expected)
    assert np.allclose(ds["I_ext"].to_numpy(), _SPEC.current())
