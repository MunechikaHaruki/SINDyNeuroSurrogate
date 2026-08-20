"""電流の単位規約の回帰テスト。

**全 kernel は電流を密度 [μA/cm^2] で受ける**。絶対量 [μA] で来るのは軸索 coupling
(g_axial·dV) だけで、その /area は kernel へ渡す値を組み立てる simulator が行う。
かつて traub kernel だけが `u_t/area` を持ち、外部注入は net 側の面積スケールで
絶対量に直してから渡していた — 型ごとに単位が違い、図の軸も I_ext の中身も
モデル依存になっていた。
"""

from dataclasses import replace as dc_replace

import numpy as np

from neurosurrogate.core import access
from neurosurrogate.core.simulator import _areas, _graph_laplacian, unified_simulator
from neurosurrogate.neurons import MCMODELS
from neurosurrogate.neurons.compartments.traub import traub_dv
from neurosurrogate.neurons.traub19 import SOMA_IDX, params_at
from neurosurrogate.sim.spec import SimSpec

_SOMA = MCMODELS["traub19"].nodes[SOMA_IDX]
_SPEC = SimSpec(
    target="traub19",
    current_type="lin&steady",
    dt=0.01,
    current_params={"silence_duration": 1.0, "duration": 5.0, "value": 5.0},
)


def test_replaced_nodes_keep_their_area():
    """置換で type が差し替わっても面積は保たれる。

    `Compartment.area` は params から解く (`getattr(params, "area", 1.0)`) ので、
    params を落とした複製を作ると黙って 1.0 = 密度化が消え、その comp への流入だけ
    桁が変わる。`Surrogate.apply` が type だけ差し替えるのはこのため。
    """
    net = MCMODELS["traub19_soma"]
    surr_type = dc_replace(_SOMA.type, name="surr", param_cls=None)
    replaced = dc_replace(net, nodes=[dc_replace(c, type=surr_type) for c in net.nodes])
    assert np.allclose(_areas(replaced), _areas(net))
    assert not np.allclose(
        _areas(dc_replace(net, nodes=[dc_replace(c, params=None) for c in net.nodes])),
        _areas(net),
    )


def test_dv_does_not_depend_on_area():
    """同じ電流密度・同じ状態なら、面積が違っても dV/dt は同じ。
    面積が dv に効いていたら kernel が密度以外の単位を受けている。"""
    states = np.array(_SOMA.init[1:])
    dv = traub_dv(_SOMA.resolved_params, 2.5, _SOMA.init[0], states)
    wide = _SOMA.resolved_params._replace(area=_SOMA.area * 10)
    assert float(traub_dv(wide, 2.5, _SOMA.init[0], states)) == float(dv)


def test_i_internal_divides_the_coupling_by_the_receiving_node_area():
    """記録される I_internal = coupling/area + 注入密度。ラプラシアン自体は
    conductance のまま (対称) で、密度化は組立側が持つ。"""
    net = MCMODELS["traub19"]
    laplacian = _graph_laplacian(net)
    assert np.allclose(laplacian, laplacian.T)
    assert np.allclose(_areas(net), [params_at(i).area for i in range(len(net.nodes))])

    ds = unified_simulator(_SPEC.materialize())
    expected = access.potential_matrix(ds) @ laplacian / _areas(net)
    expected[:, net.name_to_idx(_SPEC.stim)] += _SPEC.current()
    assert np.allclose(ds["I_internal"].to_numpy(), expected)


def test_i_ext_is_the_density_the_spec_asked_for():
    """記録される I_ext = spec の電流波形そのもの (換算を挟まない)。"""
    assert np.allclose(
        access.i_ext_values(unified_simulator(_SPEC.materialize())), _SPEC.current()
    )
