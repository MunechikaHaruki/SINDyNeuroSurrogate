"""サロゲート: 区画の状態のどの群を、何で圧縮し、何で時間発展させるか。

設定 (Config) は群のリストで、群ごとに圧縮器と右辺を持つ。群に入らなかった状態は
元のモデルの式のまま解く (hybrid)。form="sindy" は膜電位も含めて丸ごと同定する。
kernel・初期値・演算コストは、群と区画の式から組む。
"""

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import joblib
import numpy as np
import xarray as xr

from ..physics.cells import CELLS, CellType, Compartment, Net, total
from ..physics.simulate import (
    EvalSeries,
    RunInput,
    SimSpec,
    gate_matrix,
    i_internal,
    latent_names,
    potential,
    simulate,
    times,
)
from .compress import FITS
from .sindy import Roles, fit_sindy
from .ude import fit_ude

# --- 設定 ------------------------------------------------------------------------


@dataclass(frozen=True)
class Group:
    """1 つの群: 圧縮する状態と、圧縮器と右辺。params は各 fit 関数の引数へ展開する。"""

    gates: list[str]
    compressor: str  # "pca" | "ae"
    d: int  # 潜在次元
    rhs: str = "sindy"  # "sindy" | "ude"
    compressor_params: dict = field(default_factory=dict)
    rhs_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    cell: str  # 区画の型 (cells.CELLS の鍵)
    form: str  # "hybrid" | "sindy"
    data: SimSpec  # 学習データを取るシミュレーション
    groups: list[Group]
    train_comp: str | None = None  # 学習を 1 区画へ絞る名前 (None は同じ型の全区画)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        config = cls(
            cell=d["cell"],
            form=d["form"],
            data=SimSpec.from_dict(d["data"]),
            groups=[Group(**g) for g in d["groups"]],
            train_comp=d.get("train_comp"),
        )
        config._check()
        return config

    def to_dict(self) -> dict:
        return {
            "cell": self.cell,
            "form": self.form,
            "data": self.data.to_dict(),
            "groups": [g.__dict__ for g in self.groups],
            "train_comp": self.train_comp,
        }

    def _check(self) -> None:
        learned = [g for group in self.groups for g in group.gates]
        if unknown := [g for g in learned if g not in self.cell_type.states]:
            raise ValueError(f"{self.cell} に無い状態: {unknown}")
        if len(set(learned)) != len(learned):
            raise ValueError(f"群が重なっている: {learned}")
        for group in self.groups:
            if group.rhs == "ude" and group.compressor != "ae":
                raise ValueError("ude は encoder・decoder を更新するので compressor=ae")
        if self.form == "sindy" and (
            len(self.groups) != 1 or learned != self.cell_type.states
        ):
            raise ValueError("form=sindy は全状態を 1 つの群にする")

    @property
    def cell_type(self) -> CellType:
        return CELLS[self.cell]

    @property
    def physics(self) -> list[str]:
        """元の式のまま解く状態 (型の states の並び)。"""
        learned = {g for group in self.groups for g in group.gates}
        return [n for n in self.cell_type.states if n not in learned]

    def positions(self, group: Group) -> list[int]:
        """群の状態の、型の states での位置。"""
        return [self.cell_type.states.index(g) for g in group.gates]

    def train_node(self) -> Compartment:
        """params と初期値の基準にする学習区画。"""
        net = self.data.net
        if self.train_comp is not None:
            return net.nodes[net.index(self.train_comp)]
        return next(c for c in net.nodes if c.type.name == self.cell)

    def in_domain(self, comp: Compartment) -> bool:
        """comp に当ててよいか。hybrid は params を入力で受けるので型だけを見る。"""
        if comp.type.name != self.cell:
            return False
        if self.form == "hybrid":
            return True
        return comp.resolved_params == self.train_node().resolved_params

    def train_ids(self) -> list[int]:
        net = self.data.net
        if self.train_comp is not None:
            return [net.index(self.train_comp)]
        return [i for i, c in enumerate(net.nodes) if self.in_domain(c)]


# --- 学習 ------------------------------------------------------------------------


@dataclass(eq=False)
class Surrogate:
    """学習済みのサロゲート。parts は群ごとの (圧縮器, 右辺)。"""

    config: Config
    parts: list[tuple[Any, Any]]

    @cached_property
    def training_data(self) -> xr.Dataset:
        """学習データ。保存せず、設定から決定的に作り直す。"""
        return simulate(self.config.data.run_input())

    def save(self, folder: Path) -> None:
        (folder / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False)
        )
        joblib.dump(self.parts, folder / "parts.joblib")

    @classmethod
    def load(cls, folder: Path) -> "Surrogate":
        config = Config.from_dict(json.loads((folder / "config.json").read_text()))
        return cls(config, joblib.load(folder / "parts.joblib"))


@dataclass(frozen=True)
class _Inputs:
    """同定器へ渡す列。x と u は学習区画ごとの軌道 (時刻 × 列)。"""

    x_names: list[str]
    u_names: list[str]
    x: list[np.ndarray]
    u: list[np.ndarray]


def _offsets(config: Config) -> list[int]:
    """群ごとの潜在の開始位置。"""
    offsets, start = [], 0
    for group in config.groups:
        offsets.append(start)
        start += group.d
    return offsets


def _group_gates(config: Config, group: Group, data: xr.Dataset) -> list[np.ndarray]:
    """学習区画ごとの、群の状態の軌道。"""
    positions = config.positions(group)
    return [gate_matrix(data, i)[:, positions] for i in config.train_ids()]


def _inputs(
    config: Config, group: Group, compressor, data: xr.Dataset, offset: int
) -> _Inputs:
    """hybrid は x = 潜在、u = V。form=sindy は x = [V, 潜在]、u = 流入電流。"""
    ids = config.train_ids()
    latents = [compressor.encode(g) for g in _group_gates(config, group, data)]
    names = latent_names(group.d, offset)
    if config.form == "sindy":
        x = []
        for i, z in zip(ids, latents, strict=True):
            x.append(np.concatenate((potential(data, i).reshape(-1, 1), z), axis=1))
        u = [i_internal(data, i)[:, None] for i in ids]
        return _Inputs(["V", *names], ["u"], x, u)
    return _Inputs(names, ["V"], latents, [potential(data, i)[:, None] for i in ids])


def _own_rates(group: Group, params: dict) -> dict:
    """gates を書いていない項の束は、レートを群の状態のもの (N, H なら n, h) に絞る。"""
    own = "".join(g.lower() for g in group.gates)
    specs = [
        spec if "gates" in spec else {**spec, "gates": own}
        for spec in params["library_specs"]
    ]
    return {**params, "library_specs": specs}


def _fit_rhs(config: Config, group: Group, compressor, data, offset: int):
    if group.rhs == "ude":
        gates = _group_gates(config, group, data)
        volts = [potential(data, i) for i in config.train_ids()]
        return fit_ude(
            compressor,
            jnp.asarray(np.stack(gates), dtype=jnp.float32),
            jnp.asarray(np.stack(volts), dtype=jnp.float32),
            np.concatenate(gates, axis=0),
            dt=float(config.data.dt),
            d=group.d,
            **group.rhs_params,
        )
    inputs = _inputs(config, group, compressor, data, offset)
    if config.form == "sindy":
        roles = Roles(V=0, g=list(range(1, 1 + group.d)), u=1 + group.d)
    else:
        roles = Roles(V=group.d, g=list(range(group.d)))
    return fit_sindy(
        inputs.x_names,
        inputs.u_names,
        inputs.x,
        inputs.u,
        times(data),
        roles,
        **_own_rates(group, group.rhs_params),
    )


def fit(config_dict: dict) -> Surrogate:
    """設定から学習する。群ごとに、圧縮器を学習してから右辺を同定する。"""
    config = Config.from_dict(config_dict)
    data = simulate(config.data.run_input())
    parts = []
    for group, offset in zip(config.groups, _offsets(config), strict=True):
        gates = np.concatenate(_group_gates(config, group, data), axis=0)
        compressor = FITS[group.compressor](gates, group.d, **group.compressor_params)
        parts.append((compressor, _fit_rhs(config, group, compressor, data, offset)))
    surrogate = Surrogate(config, parts)
    surrogate.training_data = data
    return surrogate


# --- 置換後の区画の型 ------------------------------------------------------------


def _rhs_fn(rhs):
    """(z, V) → dz/dt。"""
    if hasattr(rhs, "rhs"):
        return rhs.rhs()
    xi = jnp.asarray(rhs.xi)
    theta = rhs.theta()
    return lambda z, v: xi @ theta(*z, v)


def surrogate_cell(s: Surrogate) -> CellType:
    """学習したものから、置換に使う区画の型を組む。"""
    config, cell = s.config, s.config.cell_type
    name = f"{cell.name}_{config.form}_surr"
    if config.form == "sindy":
        compressor, rhs = s.parts[0]
        d = config.groups[0].d
        xi, theta = jnp.asarray(rhs.xi), rhs.theta()

        def sindy_kernel(p, u, v, x):
            th = theta(v, *(x[i] for i in range(d)), u)
            return xi[0] @ th, xi[1:] @ th

        v0 = config.train_node().init[0]
        return CellType(
            name=name,
            kernel=sindy_kernel,
            # params が一致する区画にだけ当てるので、学習区画の初期値を使う
            param_cls=None,
            states=latent_names(d),
            inits=lambda _: [v0, *compressor.gate_inits],
            opcost=rhs.opcost(),
        )

    assert cell.dv is not None and cell.derivs is not None and cell.costs is not None
    dv_fn, derivs, physics = cell.dv, cell.derivs, config.physics
    blocks = []
    for (compressor, rhs), group, start in zip(
        s.parts, config.groups, _offsets(config), strict=True
    ):
        blocks.append(
            (group.gates, start, start + group.d, compressor.decode, _rhs_fn(rhs))
        )
    n_latent = sum(group.d for group in config.groups)

    def hybrid_kernel(p, u, v, x):
        s = {}
        for gates, a, b, decode, _ in blocks:
            values = decode(x[a:b])
            for j, gate in enumerate(gates):
                s[gate] = values[j]
        for i, state in enumerate(physics):
            s[state] = x[n_latent + i]
        dx = [rhs(x[a:b], v) for _, a, b, _, rhs in blocks]
        if physics:
            dx.append(jnp.stack([derivs[n](p, v, s) for n in physics]))
        return dv_fn(p, u, v, s), jnp.concatenate(dx)

    latent_inits = [z for compressor, _ in s.parts for z in compressor.gate_inits]

    def inits(p) -> list[float]:
        full = cell.inits(p)
        rest = [full[1 + cell.states.index(n)] for n in physics]
        return [full[0], *latent_inits, *rest]

    learned = total([c.opcost() + r.opcost() for c, r in s.parts])
    assert cell.dv_cost is not None
    return CellType(
        name=name,
        kernel=hybrid_kernel,
        param_cls=cell.param_cls,
        states=latent_names(n_latent) + physics,
        inits=inits,
        opcost=learned + cell.dv_cost + total([cell.costs[n] for n in physics]),
    )


# --- 置換 ------------------------------------------------------------------------


def rejections(
    config: Config, net: Net, targets: list[str] | tuple[str, ...]
) -> list[str]:
    """置き換えられない区画とその理由 (空なら全部置き換えられる)。"""
    reasons = []
    for target in targets:
        if target not in net.names:
            reasons.append(f"{target!r}: 適用先 {net.names} に存在しない")
            continue
        comp = net.nodes[net.index(target)]
        if config.in_domain(comp):
            continue
        if comp.type.name != config.cell:
            reasons.append(
                f"{target!r}: 種類 {comp.type.name!r} ≠ 学習した種類 {config.cell!r}"
            )
        else:
            train = config.train_node()
            reasons.append(
                f"{target!r}: params が学習区画 {train.name} と違う\n"
                f"    train: {train.resolved_params}\n    node: {comp.resolved_params}"
            )
    return reasons


def applicable(config: Config, series: EvalSeries) -> bool:
    """系列が挙げた区画を全部置き換えられるか。"""
    targets = series.replace_targets
    return bool(targets) and not rejections(config, series.spec.net, targets)


def replace(s: Surrogate, run: RunInput, targets) -> RunInput:
    """targets の区画の型をサロゲートに差し替えた実行入力。一部だけの置換はしない。"""
    if not targets:
        raise ValueError("置換する区画を名前で挙げる")
    if reasons := rejections(s.config, run.net, targets):
        raise ValueError("置き換えられない区画:\n  " + "\n  ".join(reasons))
    cell = surrogate_cell(s)
    nodes = []
    for node in run.net.nodes:
        nodes.append(
            Compartment(node.name, cell, node.params) if node.name in targets else node
        )
    return RunInput(run.dt, Net(nodes, run.net.edges), run.stim_idx, run.current)


# --- 中身の書き出し --------------------------------------------------------------


def summary(s: Surrogate) -> dict[str, Any]:
    """学習側の指標と、1 刻みの演算コスト (置換後・元・差)。"""
    metrics: dict[str, Any] = {}
    for (compressor, rhs), group in zip(s.parts, s.config.groups, strict=True):
        prefix = "" if len(s.parts) == 1 else "".join(group.gates).lower() + "/"
        for k, v in {**rhs.metrics(), **compressor.metrics()}.items():
            metrics[prefix + k] = v
    surr = surrogate_cell(s).opcost.to_dict()
    orig = s.config.cell_type.opcost.to_dict()
    return {
        **metrics,
        **{f"cost/surrogate/{k}": v for k, v in surr.items()},
        **{f"cost/original/{k}": v for k, v in orig.items()},
        **{f"cost/surr-orig/{k}": surr[k] - orig[k] for k in orig},
    }


@dataclass(frozen=True)
class CompFit:
    """学習区画 1 つの、同定器へ渡した列・潜在・再構成の RMSE (時刻ごと)。"""

    x: np.ndarray
    latent: np.ndarray
    recon_rmse: np.ndarray


def describe(s: Surrogate) -> tuple[dict, dict[int, CompFit]]:
    """図が読む中身 (model.json) と、学習区画ごとの CompFit。"""
    config, data = s.config, s.training_data
    net = config.data.net
    ids = config.train_ids()
    per_group = []
    for (compressor, _), group, offset in zip(
        s.parts, config.groups, _offsets(config), strict=True
    ):
        gates = _group_gates(config, group, data)
        latents = [np.asarray(compressor.encode(g)) for g in gates]
        recons = [np.asarray(compressor.decode(jnp.asarray(z))) for z in latents]
        inputs = _inputs(config, group, compressor, data, offset)
        per_group.append((gates, latents, recons, inputs))
    fits = {}
    for pos, comp_id in enumerate(ids):
        sq = np.concatenate(
            [(g[pos] - r[pos]) ** 2 for g, _, r, _ in per_group], axis=1
        )
        fits[comp_id] = CompFit(
            x=np.concatenate([inp.x[pos] for _, _, _, inp in per_group], axis=1),
            latent=np.concatenate([z[pos] for _, z, _, _ in per_group], axis=1),
            recon_rmse=np.sqrt(np.mean(sq, axis=1)),
        )
    first_compressor, first_rhs = s.parts[0]
    model: dict[str, Any] = {
        "closure": first_rhs.describe(),
        "preprocessor": first_compressor.describe(),
        "net": {
            "nodes": [{"name": c.name, "type": c.type.name} for c in net.nodes],
            "edges": [
                {"src": e.src, "dst": e.dst, "weight": e.weight} for e in net.edges
            ],
            "stim": config.data.stim,
            "replaceable": [c.name for c in net.nodes if config.in_domain(c)],
            "train": [net.nodes[i].name for i in ids],
        },
        "train_gates": [g for group in config.groups for g in group.gates],
        "x_names": [n for _, _, _, inp in per_group for n in inp.x_names],
        "latent_names": latent_names(sum(group.d for group in config.groups)),
    }
    if len(s.parts) > 1:
        model["groups"] = [
            {
                "gates": group.gates,
                "closure": r.describe(),
                "preprocessor": c.describe(),
            }
            for (c, r), group in zip(s.parts, config.groups, strict=True)
        ]
    return model, fits
