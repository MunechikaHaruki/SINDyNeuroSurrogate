"""ネットを解くシミュレータと、波形 (xarray) の形と、実験の記述 (SimSpec・EvalSeries)。

シミュレータは前進 Euler (JAX) の simulate。

波形の Dataset は vars (time × features) と I_ext、I_internal を持つ。features は
(comp_id, variable, gate) の MultiIndex で、gate=False が電位。並びは全区画の電位の
あとに、区画ごとの状態が続く。
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from .cells import NETS, Net
from .currents import CURRENTS, PARAM_UNITS

# --- 実行入力 --------------------------------------------------------------------


@dataclass
class RunInput:
    """解く対象: ネット、刻み、注入する区画、注入電流密度 [μA/cm^2]。"""

    dt: float
    net: Net
    stim_idx: int
    current: np.ndarray


def _laplacian(net: Net) -> np.ndarray:
    """V @ L が各区画への流入電流 [μA] になる対称ラプラシアン。"""
    n = len(net.nodes)
    g = np.zeros((n, n), dtype=np.float64)
    for e in net.edges:
        i, j = net.index(e.src), net.index(e.dst)
        g[i, j] = g[j, i] = e.weight
    return g - np.diag(np.sum(g, axis=1))


def _areas(net: Net) -> np.ndarray:
    return np.array([c.area for c in net.nodes])


@dataclass(frozen=True)
class _Group:
    """同じ型の区画をまとめて vmap で解く単位。"""

    kernel: Any
    n_state: int
    indices: np.ndarray
    params: Any  # 区画ごとの params を束ねた NamedTuple (param_cls が無ければ None)


def _groups(net: Net) -> list[_Group]:
    buckets: dict[str, list[int]] = {}
    for i, comp in enumerate(net.nodes):
        buckets.setdefault(comp.type.name, []).append(i)
    groups = []
    for indices in buckets.values():
        comps = [net.nodes[i] for i in indices]
        cell = comps[0].type
        params = None
        if cell.param_cls is not None:
            params = jax.tree.map(
                lambda *xs: jnp.asarray(xs), *[c.resolved_params for c in comps]
            )
        groups.append(
            _Group(cell.kernel, len(cell.states), np.array(indices, np.int32), params)
        )
    return groups


def _layout(net: Net) -> tuple[pd.MultiIndex, np.ndarray, np.ndarray]:
    """features の座標、初期値、区画ごとの状態の開始位置。"""
    n = len(net.nodes)
    comp_ids, names, gates, inits = [], [], [], []
    for i, comp in enumerate(net.nodes):
        comp_ids.append(i)
        names.append("V")
        gates.append(False)
        inits.append(comp.init[0])
    offsets = np.full(n, -1, dtype=np.int32)
    offset = n
    for i, comp in enumerate(net.nodes):
        states = comp.type.states
        if states:
            offsets[i] = offset
            comp_ids.extend([i] * len(states))
            names.extend(states)
            gates.extend([True] * len(states))
            inits.extend(comp.init[1:])
            offset += len(states)
    index = pd.MultiIndex.from_arrays(
        [comp_ids, names, gates], names=("comp_id", "variable", "gate")
    )
    return index, np.array(inits, dtype=np.float64), offsets


def _euler(init, current, dt, laplacian, areas, stim_idx, offsets, groups):
    n = laplacian.shape[0]

    def deriv(x, u_t):
        v = x[:n]
        # kernel は電流密度を受ける: coupling を面積で割り、密度の注入を足す
        i_in = (v @ laplacian / areas).at[stim_idx].add(u_t)
        dx = jnp.zeros_like(x)
        for g in groups:
            idx = g.indices
            state_idx = offsets[idx][:, None] + np.arange(g.n_state)
            in_axes = (None if g.params is None else 0, 0, 0, 0)
            dv, ds = jax.vmap(g.kernel, in_axes=in_axes)(
                g.params, i_in[idx], v[idx], x[state_idx]
            )
            dx = dx.at[idx].set(dv)
            if g.n_state > 0:
                dx = dx.at[state_idx.ravel()].set(ds.ravel())
        return dx

    def step(x, u_t):
        return x + deriv(x, u_t) * dt, x

    final, history = jax.lax.scan(step, jnp.array(init), jnp.array(current)[:-1])
    return np.array(jnp.concatenate([history, final[None]], axis=0))


def simulate(run: RunInput) -> xr.Dataset:
    """前進 Euler で解いた波形。"""
    index, init, offsets = _layout(run.net)
    laplacian, areas = _laplacian(run.net), _areas(run.net)
    raw = _euler(
        init,
        run.current,
        run.dt,
        laplacian,
        areas,
        run.stim_idx,
        offsets,
        _groups(run.net),
    )
    ds = waves_dataset(raw, run.current, index, run.dt)
    i_ext = np.zeros((len(run.current), len(run.net.nodes)), dtype=np.float64)
    i_ext[:, run.stim_idx] = run.current
    # シミュレータと同じ組み立て (coupling を面積で割って密度へ)
    i_internal = potential_matrix(ds) @ laplacian / areas + i_ext
    ds["I_internal"] = xr.DataArray(
        i_internal,
        coords={"time": ds.time, "node_id": np.arange(i_internal.shape[1])},
        dims=["time", "node_id"],
    )
    return ds


# --- 波形の読み書き -------------------------------------------------------------


def latent_names(n: int, start: int = 0) -> list[str]:
    """潜在変数の名前 z1..zN (start で番号をずらす)。"""
    return [f"z{start + i + 1}" for i in range(n)]


def waves_dataset(raw, current, index: pd.MultiIndex, dt: float) -> xr.Dataset:
    return xr.Dataset(
        {"vars": (("time", "features"), raw), "I_ext": (("time"), current)},
        coords={
            "time": np.arange(len(current)) * dt,
            **xr.Coordinates.from_pandas_multiindex(index, "features"),
        },
    )


def latent_dataset(
    v: np.ndarray, latent: np.ndarray, u: np.ndarray, comp_id: int, dt: float
) -> xr.Dataset:
    """1 区画の [V, z1..zN] と流入電流 u の Dataset。"""
    n = latent.shape[1]
    index = pd.MultiIndex.from_arrays(
        [[comp_id] * (n + 1), ["V", *latent_names(n)], [False] + [True] * n],
        names=("comp_id", "variable", "gate"),
    )
    raw = np.concatenate((v.reshape(-1, 1), latent), axis=1)
    return waves_dataset(raw, u, index, dt)


def times(ds: xr.Dataset) -> np.ndarray:
    return ds["time"].values


def comp_ids(ds: xr.Dataset) -> np.ndarray:
    return np.unique(ds.coords["comp_id"].values)


def potential(ds: xr.Dataset, comp_id: int) -> np.ndarray:
    return ds["vars"].sel(gate=False, comp_id=comp_id).to_numpy().squeeze()


def gate_matrix(ds: xr.Dataset, comp_id: int) -> np.ndarray:
    """区画の状態 (time × 状態)。並びは型の states。"""
    return ds["vars"].sel(gate=True, comp_id=comp_id).to_numpy()


def potential_matrix(ds: xr.Dataset) -> np.ndarray:
    """全区画の電位 (time × 区画)。"""
    return ds["vars"].sel(gate=False).sortby("comp_id").to_numpy()


def i_internal(ds: xr.Dataset, comp_id: int) -> np.ndarray:
    return ds["I_internal"].sel(node_id=comp_id).to_numpy()


# --- 実験の記述 ------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SimSpec:
    """1 回のシミュレーション: 適用先 × 電流。これだけで波形が決まる。"""

    target: str
    current_type: str
    dt: float
    current_params: dict = field(default_factory=dict)
    stim: str = "soma"  # 注入する区画の名前

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "current_type": self.current_type,
            "dt": self.dt,
            "current_params": self.current_params,
            "stim": self.stim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            target=str(d["target"]),
            current_type=str(d["current_type"]),
            dt=float(d["dt"]),
            current_params=dict(d.get("current_params") or {}),
            stim=str(d.get("stim", "soma")),
        )

    @property
    def net(self) -> Net:
        return NETS[self.target]

    def current(self) -> np.ndarray:
        return CURRENTS[self.current_type](**self.current_params)(self.dt)

    def run_input(self) -> RunInput:
        return RunInput(self.dt, self.net, self.net.index(self.stim), self.current())


@dataclass(frozen=True)
class EvalSeries:
    """1 回の掃引: 何を、どこを置き換えて、どの電流パラメータで振るか。"""

    spec: SimSpec
    replace_targets: tuple[str, ...]
    param: str | None = None  # 掃引する電流パラメータ (None は単発)
    values: Sequence[float] = ()

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            spec=SimSpec.from_dict(d["spec"]),
            replace_targets=tuple(d["replace_targets"]),
            param=d["param"] or None,
            values=[float(v) for v in d["values"]],
        )

    @property
    def points(self) -> list[SimSpec]:
        """点ごとの計算入力 (current_params[param] に値を埋めたもの)。"""
        if self.param is None:
            return [self.spec]
        points = []
        for v in self.values:
            params = {**self.spec.current_params, self.param: float(v)}
            points.append(dc_replace(self.spec, current_params=params))
        return points

    @property
    def unit(self) -> str:
        return PARAM_UNITS.get(self.param or "", "")
