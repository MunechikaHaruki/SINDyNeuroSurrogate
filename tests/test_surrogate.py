"""サロゲートの学習・置換・書き出し。preset と系列は conf.ts の宣言を段取りから読む。"""

import json
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from sympy.core.function import AppliedUndef

from neurosurrogate.physics.cells import CELLS, NETS, OpCost
from neurosurrogate.physics.simulate import EvalSeries, SimSpec, potential, simulate
from neurosurrogate.spikes import FEATURES, spike_features
from neurosurrogate.surrogate.compress import AE, fit_ae
from neurosurrogate.surrogate.model import (
    Config,
    Surrogate,
    applicable,
    describe,
    fit,
    rejections,
    replace,
    summary,
    surrogate_cell,
)
from neurosurrogate.surrogate.sindy import SINDy
from neurosurrogate.surrogate.ude import UDE

ORCHESTRATE = (
    Path(__file__).resolve().parents[1] / "evaluation/orchestrate/orchestrate.ts"
)


def _declared(*args: str) -> Any:
    return json.loads(subprocess.check_output(["bun", str(ORCHESTRATE), *args]))


@cache
def preset_config(preset: str) -> dict:
    """preset の先頭の学習 run の設定。"""
    config: dict = _declared("spec", preset)[0]
    return config


@cache
def fit_preset(preset: str) -> Surrogate:
    """テスト専用の preset を学習する (決定的なので使い回す)。"""
    return fit(preset_config(preset))


def _dc(target: str, value: float = 3.0, duration: float = 40.0) -> SimSpec:
    return SimSpec(
        target=target,
        current_type="lin&steady",
        dt=0.01,
        current_params={"silence_duration": 10.0, "duration": duration, "value": value},
    )


# --- 恒等サロゲート: 圧縮を恒等、右辺を真の式にすると元のモデルと一致する --------


class _Identity:
    """encode・decode が素通しの圧縮器。"""

    def __init__(self, gate_inits: list[float]) -> None:
        self.gate_inits = gate_inits

    def encode(self, x):
        return x

    def decode(self, z):
        return z

    def opcost(self) -> OpCost:
        return OpCost()


class _Exact:
    """群のゲートの真の微分を返す右辺。ゲートは自分の値と V だけで動くものに限る。"""

    def __init__(self, cell: str, gates: list[str], params) -> None:
        derivs = CELLS[cell].derivs
        assert derivs is not None
        self.derivs = [derivs[g] for g in gates]
        self.gates = gates
        self.params = params

    def rhs(self):
        def dz(z, v):
            s = {g: z[j] for j, g in enumerate(self.gates)}
            return jnp.stack([d(self.params, v, s) for d in self.derivs])

        return dz

    def opcost(self) -> OpCost:
        return OpCost()


def _identity_surrogate(cell: str, groups: list[list[str]]) -> Surrogate:
    data = {"target": cell, "current_type": "train", "dt": 0.01}
    config = Config.from_dict(
        {
            "cell": cell,
            "form": "hybrid",
            "data": data,
            "groups": [{"gates": g, "compressor": "pca", "d": len(g)} for g in groups],
        }
    )
    node = NETS[cell].nodes[0]
    init = node.init
    parts = []
    for gates in groups:
        inits = [init[1 + CELLS[cell].states.index(g)] for g in gates]
        parts.append((_Identity(inits), _Exact(cell, gates, node.resolved_params)))
    return Surrogate(config, parts)


@pytest.mark.parametrize(
    ("cell", "groups"),
    [
        ("hh", [["M", "H", "N"]]),
        ("hh", [["H"], ["N", "M"]]),
        ("traub", [["M", "N", "C", "A", "H", "B", "S", "R"]]),
        ("traub", [["M", "N", "C", "A", "H", "B"]]),
        ("traub", [["N", "H"]]),
        ("traub", [["N", "H"], ["A", "C"], ["B"]]),
    ],
)
def test_any_grouping_with_exact_parts_reproduces_the_original(cell, groups) -> None:
    """どんな群分けでも、恒等の圧縮と真の右辺なら元のモデルと同じ電位になる。
    群の並べ直しと、元の式で解く状態の組み立てにずれが無いことの担保。"""
    s = _identity_surrogate(cell, groups)
    spec = _dc(cell, value=10.0)
    original = potential(simulate(spec.run_input()), 0)
    replaced = potential(simulate(replace(s, spec.run_input(), ["soma"])), 0)
    np.testing.assert_allclose(replaced, original, rtol=0, atol=1e-9)


# --- 宣言した preset の学習 ------------------------------------------------------


@pytest.mark.parametrize(
    "preset",
    [
        "_test_hh_sindy",
        "_test_hh_hybrid",
        "_test_hh_ae",
        "_test_traub_hybrid",
        "_test_traub_hybrid_sr_physics",
        "_test_traub_hybrid_hn",
        "_test_traub_two_groups",
        "_test_traub_kam",
    ],
)
def test_presets_fit_and_replace_finitely(preset: str) -> None:
    """学習した入力で置き換えて回すと有限に走る (短い学習の質までは見ない)。"""
    s = fit_preset(preset)
    run = replace(s, s.config.data.run_input(), ["soma"])
    assert np.isfinite(potential(simulate(run), 0)).all()
    cell = surrogate_cell(s)
    n_latent = sum(group.d for group in s.config.groups)
    assert cell.states == [f"z{i + 1}" for i in range(n_latent)] + (
        [] if s.config.form == "sindy" else s.config.physics
    )


def test_library_gates_keep_only_those_rates() -> None:
    rhs = fit_preset("_test_traub_hybrid_hn").parts[0][1]
    assert isinstance(rhs, SINDy)
    rates = {str(f.func) for e in rhs.feature_exprs for f in e.atoms(AppliedUndef)}
    assert rates == {"alpha_n_traub", "beta_n_traub", "alpha_h_traub", "beta_h_traub"}


def test_each_group_keeps_only_its_own_rates() -> None:
    """gates を書かなければ、群の右辺のレート項は群の状態のものだけになる。"""
    (_, first), (_, second) = fit_preset("_test_traub_two_groups").parts
    for rhs, gates in [(first, "nh"), (second, "ac")]:
        assert isinstance(rhs, SINDy)
        rates = {str(f.func) for e in rhs.feature_exprs for f in e.atoms(AppliedUndef)}
        assert rates == {f"{k}_{g}_traub" for g in gates for k in ["alpha", "beta"]}


def test_two_groups_keep_their_latents_apart() -> None:
    """群の右辺は自分の潜在と V だけを使い、潜在の名前は通し番号。"""
    s = fit_preset("_test_traub_two_groups")
    (_, first), (_, second) = s.parts
    assert [str(t) for t in first.targets] == ["z1"]
    assert [str(t) for t in second.targets] == ["z2"]
    assert all("z2" not in str(e) for e in first.feature_exprs)
    assert s.config.physics == ["M", "B", "S", "R", "Q", "XI"]


def test_kam_keeps_the_rest_state_and_the_spike_count() -> None:
    """KAM は静止状態で ψ = V かつ動かず、DC 入力で元のモデルと同じ回数発火する。"""
    s = fit_preset("_test_traub_kam")
    spec = _dc("traub", value=5.0, duration=100.0)
    rest = CELLS["traub"].inits(spec.net.nodes[0].resolved_params)
    for (compressor, rhs), group in zip(s.parts, s.config.groups, strict=True):
        assert compressor.gate_inits == pytest.approx([rest[0]], abs=1e-6)
        state = dict(zip(CELLS["traub"].states, rest[1:], strict=True))
        p = spec.net.nodes[0].resolved_params
        assert float(rhs.rhs()(jnp.asarray([rest[0]]), rest[0], state, p)[0]) == (
            pytest.approx(0.0, abs=1e-9)
        )
        gates = np.array([[state[g] for g in group.gates]])
        decoded = np.asarray(compressor.decode(jnp.asarray(compressor.encode(gates))))
        assert decoded == pytest.approx(gates, abs=1e-6)

    def count(v: np.ndarray) -> int:
        return int(np.sum((v[:-1] < 0) & (v[1:] >= 0)))

    original = potential(simulate(spec.run_input()), 0)
    reduced = potential(simulate(replace(s, spec.run_input(), ["soma"])), 0)
    assert count(original) > 0 and count(reduced) == count(original)


def test_config_rejects_bad_groups() -> None:
    base = preset_config("_test_traub_hybrid")
    group = base["groups"][0]
    for groups, message in [
        ([{**group, "gates": ["X"]}], "無い状態"),
        ([{**group, "gates": ["N"]}, {**group, "gates": ["N", "H"]}], "重なって"),
        ([{**group, "rhs": "ude"}], "compressor=ae"),
        ([{**group, "compressor": "kam"}], "両方を kam"),
        (
            [{**group, "gates": ["N"], "compressor": "kam", "rhs": "kam", "d": 1}],
            "2 つ以上",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            Config.from_dict({**base, "groups": groups})
    with pytest.raises(ValueError, match="全状態"):
        Config.from_dict({**base, "form": "sindy"})


def test_ude_updates_the_autoencoder() -> None:
    s = fit_preset("_test_traub_ude")
    ae, rhs = s.parts[0]
    assert isinstance(ae, AE) and isinstance(rhs, UDE)
    gates = s.training_data["vars"].sel(gate=True, comp_id=0).to_numpy()[:, :8]
    before = fit_ae(gates, 3, epochs=20)
    assert not np.allclose(ae.dec_params["W2"], before.dec_params["W2"])
    assert len(ae.gate_inits) == 3


# --- 置換の可否 -----------------------------------------------------------------


def test_hybrid_transplants_to_every_traub19_compartment() -> None:
    """hybrid は params を入力で受けるので、密度の違う全区画へ当てられる。"""
    s = fit_preset("_test_traub_hybrid_hn")
    net = NETS["traub19"]
    assert rejections(s.config, net, net.names) == []
    spec = SimSpec(target="traub19", current_type="train", dt=0.01)
    v = potential(simulate(replace(s, spec.run_input(), net.names)), 8)
    assert np.isfinite(v).all()


def test_rejections_name_each_target_and_reason() -> None:
    sindy = fit_preset("_test_hh_sindy")  # form=sindy は params が一致する区画だけ
    traub = fit_preset("_test_traub_hybrid")
    assert "存在しない" in rejections(sindy.config, NETS["hh"], ["nowhere"])[0]
    assert "種類" in rejections(traub.config, NETS["hh"], ["soma"])[0]
    hh_series = EvalSeries(_dc("hh"), ("soma",))
    assert applicable(sindy.config, hh_series)
    assert not applicable(sindy.config, EvalSeries(_dc("hh"), ()))
    with pytest.raises(ValueError, match="名前で挙げる"):
        replace(sindy, _dc("hh").run_input(), [])


def test_replace_swaps_only_the_named_compartments() -> None:
    s = fit_preset("_test_traub_hybrid")
    run = replace(
        s,
        SimSpec(target="traub19", current_type="train", dt=0.01).run_input(),
        ["soma", "c09"],
    )
    swapped = [c.name for c in run.net.nodes if c.type.name != "traub"]
    assert swapped == ["soma", "c09"]
    assert [c.area for c in run.net.nodes] == [c.area for c in NETS["traub19"].nodes]


# --- 書き出し -------------------------------------------------------------------


def test_description_survives_reload_and_lines_up(tmp_path: Path) -> None:
    s = fit_preset("_test_hh_sindy")
    s.save(tmp_path)
    reloaded = Surrogate.load(tmp_path)
    assert reloaded.config == s.config
    model, fits = describe(reloaded)
    json.dumps(model)
    assert model["closure"]["kind"] == "sindy"
    assert np.shape(model["closure"]["xi"]) == (
        len(model["closure"]["targets"]),
        len(model["closure"]["features"]),
    )
    n_time = reloaded.training_data.sizes["time"]
    comp_fit = fits[0]
    assert comp_fit.x.shape == (n_time, len(model["x_names"]))
    assert comp_fit.latent.shape == (n_time, len(model["latent_names"]))
    compressor = s.parts[0][0]
    assert float(np.mean(comp_fit.recon_rmse**2)) == pytest.approx(
        compressor.reconstruction["reconstruction_mse"]
    )


def test_two_groups_describe_every_group() -> None:
    model, fits = describe(fit_preset("_test_traub_two_groups"))
    assert [g["gates"] for g in model["groups"]] == [["N", "H"], ["A", "C"]]
    assert model["x_names"] == model["latent_names"] == ["z1", "z2"]
    assert fits[0].latent.shape[1] == 2


def test_summary_counts_the_decode_and_the_physics() -> None:
    s = fit_preset("_test_traub_hybrid_hn")
    result = summary(s)
    assert {"nnz", "pca/reconstruction_mse", "cost/surr-orig/mul"} <= set(result)
    compressor, rhs = s.parts[0]
    cell = CELLS["traub"]
    assert cell.costs is not None and cell.dv_cost is not None
    expected = compressor.opcost() + rhs.opcost() + cell.dv_cost
    for state in s.config.physics:
        expected = expected + cell.costs[state]
    assert surrogate_cell(s).opcost == expected


def test_spike_features_count_the_spikes_of_one_trace() -> None:
    spec = _dc("hh", value=10.0)
    v = potential(simulate(spec.run_input()), 0)
    features = spike_features(v, spec.dt)
    assert set(features) == set(FEATURES)
    peaks = features["peak_indices"]
    assert len(peaks) > 0 and len(features["peak_voltage"]) == len(peaks)
    assert spike_features(np.full_like(v, -65.0), spec.dt)["peak_indices"] == []


def test_declared_series_resolve() -> None:
    """宣言した評価系列は、電流と置換する区画まで Python の名前で解ける。"""
    for d in _declared("series").values():
        series = EvalSeries.from_dict(d)
        for point in series.points:
            assert len(point.current()) > 0
        assert set(series.replace_targets) <= set(series.spec.net.names)
