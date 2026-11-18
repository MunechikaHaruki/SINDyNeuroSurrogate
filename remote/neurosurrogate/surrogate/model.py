"""サロゲート: 区画の状態のどの群を、何で圧縮し、何で時間発展させるか。

設定 (Config) は群のリストで、群ごとに圧縮器と右辺を持つ。群に入らなかった状態は
元のモデルの式のまま解く (hybrid)。form="sindy" は膜電位も含めて丸ごと同定する。
kernel・初期値・演算コストは、群と区画の式から組む。学習は群 1 つ (学習器) ごとに
行い (fit_group)、学習器の並びをサロゲートに組む。置換に使う区画の型 (surrogate_cell)、
置換の可否と置換、学習器とサロゲートの中身の書き出しもここが持つ。
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from ..physics.cells import CELLS, CellType, ColumnCell, Compartment, Net, total
from ..physics.kam import KamCompressor, KamRhs, absorbed, absorbed_cost
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
    compressor: str  # "pca" | "ae" | "kam" (学習しない KAM 縮約。rhs も "kam")
    d: int  # 潜在次元
    rhs: str = "sindy"  # "sindy" | "ude" | "kam"
    compressor_params: dict = field(default_factory=dict)
    rhs_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    cell: str  # 区画の型 (cells.CELLS の鍵)
    form: str  # "hybrid" | "sindy"
    data: SimSpec | None  # 学習データを取るシミュレーション (群が KAM だけなら無い)
    groups: list[Group]
    train_comp: str | None = None  # 学習を 1 区画へ絞る名前 (None は同じ型の全区画)
    # 膜電位へ吸収するゲート (KAM の φ 群)。V での定常値に置き、状態を持たない
    absorbed: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        config = cls(
            cell=d["cell"],
            form=d["form"],
            data=SimSpec.from_dict(d["data"]) if d.get("data") is not None else None,
            groups=[Group(**g) for g in d["groups"]],
            train_comp=d.get("train_comp"),
            absorbed=d.get("absorbed", []),
        )
        config._check()
        return config

    def to_dict(self) -> dict:
        return {
            "cell": self.cell,
            "form": self.form,
            "data": self.data.to_dict() if self.data is not None else None,
            "groups": [g.__dict__ for g in self.groups],
            "train_comp": self.train_comp,
            **({"absorbed": self.absorbed} if self.absorbed else {}),
        }

    def _check(self) -> None:
        learned = [g for group in self.groups for g in group.gates]
        removed = learned + self.absorbed
        if unknown := [g for g in removed if g not in self.cell_type.states]:
            raise ValueError(f"{self.cell} に無い状態: {unknown}")
        if len(set(removed)) != len(removed):
            raise ValueError(f"群と吸収が重なっている: {removed}")
        voltage = self.cell_type.rates or {}
        if outside := [g for g in self.absorbed if g not in voltage]:
            raise ValueError(f"電位だけで動くゲートでないと吸収できない: {outside}")
        if self.absorbed and self.form != "hybrid":
            raise ValueError("膜電位への吸収は hybrid だけ")
        for group in self.groups:
            if group.rhs == "ude" and group.compressor != "ae":
                raise ValueError("ude は encoder・decoder を更新するので compressor=ae")
            kam = group.compressor == "kam"
            if kam != (group.rhs == "kam"):
                raise ValueError("kam は圧縮器と右辺の両方を kam にする")
            if kam and (group.d != 1 or len(group.gates) < 2 or self.form != "hybrid"):
                raise ValueError("kam は 2 つ以上のゲートを d=1 の ψ にまとめる hybrid")
            if not kam and self.data is None:
                raise ValueError("KAM でない群は学習データから学習する")
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
        removed = {g for group in self.groups for g in group.gates} | set(self.absorbed)
        return [n for n in self.cell_type.states if n not in removed]

    def positions(self, group: Group) -> list[int]:
        """群の状態の、型の states での位置。"""
        return [self.cell_type.states.index(g) for g in group.gates]

    def _net(self) -> Net:
        """学習データのネット。群が KAM だけの設定は学習データも学習区画も持たない。"""
        if self.data is None:
            raise ValueError("学習データの無い設定は学習区画を持たない")
        return self.data.net

    def train_node(self) -> Compartment:
        """params と初期値の基準にする学習区画。"""
        net = self._net()
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
        net = self._net()
        if self.train_comp is not None:
            return [net.index(self.train_comp)]
        return [i for i, c in enumerate(net.nodes) if self.in_domain(c)]


# --- 学習 ------------------------------------------------------------------------


@dataclass(eq=False)
class Surrogate:
    """学習器の並びを組んだサロゲート。parts は群ごとの学習器 (圧縮器, 右辺)。"""

    config: Config
    parts: list[tuple[Any, Any]]

    @cached_property
    def training_data(self) -> xr.Dataset:
        """学習データ。設定から決定的に作り直す。"""
        if self.config.data is None:
            raise ValueError("群が KAM だけのサロゲートは学習データを持たない")
        return simulate(self.config.data.run_input())


@dataclass(frozen=True)
class _Inputs:
    """同定器へ渡す列。x と u は学習区画ごとの軌道 (時刻 × 列)。"""

    x_names: list[str]
    u_names: list[str]
    x: list[np.ndarray]
    u: list[np.ndarray]


def group_label(group: Group) -> str:
    """群の名前。状態を小文字で + につなぐ (h+n)。"""
    return "+".join(g.lower() for g in group.gates)


def group_latents(group: Group) -> list[str]:
    """群の潜在の名前。群の名前で修飾し、学習器の中で z1 から (h+n.z1)。

    どのモデルに組んでも、学習器の中身でもシミュレーションの波形でも同じ名前になる。
    """
    return [f"{group_label(group)}.{z}" for z in latent_names(group.d)]


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


def _inputs(config: Config, group: Group, compressor, data: xr.Dataset) -> _Inputs:
    """hybrid は x = 潜在、u = V。form=sindy は x = [V, 潜在]、u = 流入電流。

    潜在の名前は学習器の中で z1 から付ける。
    """
    ids = config.train_ids()
    latents = [compressor.encode(g) for g in _group_gates(config, group, data)]
    names = latent_names(group.d)
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


def _fit_rhs(config: Config, group: Group, compressor, data):
    if group.rhs == "ude":
        gates = _group_gates(config, group, data)
        volts = [potential(data, i) for i in config.train_ids()]
        assert config.data is not None
        return fit_ude(
            compressor,
            jnp.asarray(np.stack(gates), dtype=jnp.float32),
            jnp.asarray(np.stack(volts), dtype=jnp.float32),
            np.concatenate(gates, axis=0),
            dt=float(config.data.dt),
            d=group.d,
            **group.rhs_params,
        )
    inputs = _inputs(config, group, compressor, data)
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


def fit_group(config: Config, group: Group, data: xr.Dataset | None) -> tuple[Any, Any]:
    """群 1 つの学習器 (圧縮器, 右辺) を学習する。

    圧縮器を学習してから右辺を同定し、ほかの群には依らない。KAM は学習しないので
    学習データを読まず、区画の型の既定の params で組む。
    """
    if group.compressor == "kam":
        params = config.cell_type.param_cls()  # type: ignore[misc]
        kam = KamCompressor(config.cell, group.gates, params)
        return kam, KamRhs(config.cell, group.gates)
    if data is None:
        raise ValueError(f"{group.compressor} の群は学習データから学習する")
    gates = np.concatenate(_group_gates(config, group, data), axis=0)
    compressor = FITS[group.compressor](gates, group.d, **group.compressor_params)
    return compressor, _fit_rhs(config, group, compressor, data)


def fit(config_dict: dict) -> Surrogate:
    """設定から、群ごとの学習器を学習してサロゲートに組む。

    run の置き場を通す本番 (学習器ごとの run を組む) と同じく、学習器はその群だけの
    設定で学習する。
    """
    config = Config.from_dict(config_dict)
    data = simulate(config.data.run_input()) if config.data is not None else None
    alone = [
        Config.from_dict({**config_dict, "groups": [group.__dict__], "absorbed": []})
        for group in config.groups
    ]
    parts = [fit_group(c, c.groups[0], data) for c in alone]
    surrogate = Surrogate(config, parts)
    if data is not None:
        surrogate.training_data = data
    return surrogate


# --- 置換後の区画の型 ------------------------------------------------------------


def _rhs_fn(rhs):
    """(z, V, 状態の名前 → 値, params) → dz/dt。KAM だけが状態と params を読む。"""
    if isinstance(rhs, KamRhs):
        return rhs.rhs()
    if hasattr(rhs, "rhs"):
        f = rhs.rhs()
        return lambda z, v, s, p: f(z, v)
    xi = jnp.asarray(rhs.xi)
    theta = rhs.theta()
    return lambda z, v, s, p: xi @ theta(*z, v)


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
            states=group_latents(config.groups[0]),
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
    name_of, phi = cell.name, config.absorbed

    def hybrid_kernel(p, u, v, x):
        s = absorbed(name_of, phi, p, v)
        for gates, a, b, decode, _ in blocks:
            values = decode(x[a:b])
            for j, gate in enumerate(gates):
                s[gate] = values[j]
        for i, state in enumerate(physics):
            s[state] = x[n_latent + i]
        dx = [rhs(x[a:b], v, s, p) for _, a, b, _, rhs in blocks]
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
        states=[z for group in config.groups for z in group_latents(group)] + physics,
        inits=inits,
        opcost=learned
        + cell.dv_cost
        + total([cell.costs[n] for n in physics])
        + absorbed_cost(name_of, phi),
    )


def surrogate_column_cell(s: Surrogate, comp: Compartment) -> ColumnCell:
    """置き換えた区画を GPU 用にする。式は surrogate_cell の hybrid と同じ。"""
    config, cell = s.config, s.config.cell_type
    if config.form != "hybrid":
        raise NotImplementedError("GPU のカーネルは hybrid の形だけ")
    assert cell.dv is not None and cell.derivs is not None
    dv_fn, derivs, physics = cell.dv, cell.derivs, config.physics
    names = surrogate_cell(s).states
    blocks = []
    for (compressor, rhs), group, start in zip(
        s.parts, config.groups, _offsets(config), strict=True
    ):
        latent = names[start : start + group.d]
        if not hasattr(compressor, "decode_columns"):
            name = type(compressor).__name__
            raise NotImplementedError(f"{name} の圧縮器は GPU のカーネルに載らない")
        decode, dz = compressor.decode_columns, _rhs_columns(rhs)
        blocks.append((group.gates, latent, decode, dz))
    name_of, phi = cell.name, config.absorbed

    def kernel(p, u, v, st):
        s_ = absorbed(name_of, phi, p, v)
        for gates, latent, decode, _ in blocks:
            for gate, value in zip(gates, decode([st[n] for n in latent]), strict=True):
                s_[gate] = value
        for state in physics:
            s_[state] = st[state]
        ds = {}
        for _, latent, _, rhs in blocks:
            dz = rhs([st[n] for n in latent], v, s_, p)
            for n, value in zip(latent, dz, strict=True):
                ds[n] = value
        for state in physics:
            ds[state] = derivs[state](p, v, s_)
        return dv_fn(p, u, v, s_), ds

    n_latent = sum(group.d for group in config.groups)
    return ColumnCell(
        names,
        list(comp.init),
        comp.resolved_params,
        kernel,
        names[:n_latent],
        [gate for group in config.groups for gate in group.gates],
    )


def _rhs_columns(rhs):
    """群の右辺を、潜在の値の並び → dz/dt の並び にする (配列にまとめない)。"""
    if isinstance(rhs, KamRhs):
        return rhs.rhs_columns()
    if not hasattr(rhs, "xi"):
        name = type(rhs).__name__
        raise NotImplementedError(f"{name} の右辺は GPU のカーネルに載らない")
    xi, terms = np.asarray(rhs.xi), rhs.terms()

    def dz(z, v, s, p):
        values = terms(*z, v)
        out = []
        for k in range(xi.shape[0]):
            total_k = 0.0
            for t, value in enumerate(values):
                if xi[k, t] != 0:
                    total_k = total_k + float(xi[k, t]) * value
            out.append(total_k)
        return out

    return dz


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
    """組んだモデルの 1 刻みの演算コスト (置換後・元・差)。

    学習器の指標は describe_group が持つ。
    """
    surr = surrogate_cell(s).opcost.to_dict()
    orig = s.config.cell_type.opcost.to_dict()
    return {
        **{f"cost/surrogate/{k}": v for k, v in surr.items()},
        **{f"cost/original/{k}": v for k, v in orig.items()},
        **{f"cost/surr-orig/{k}": surr[k] - orig[k] for k in orig},
    }


_SLOPE_CHUNK = 100_000  # 右辺を一度に評価する時刻の数 (メモリを抑える)


def latent_slopes(
    s: Surrogate, ds: xr.Dataset, comp_id: int, params: Any
) -> tuple[np.ndarray, np.ndarray] | None:
    """元のモデルの区画 comp_id の軌道の上の、潜在の傾き 2 つ (どちらも時刻 × 潜在)。

    正解は各群の圧縮器で写した潜在の前進差分 (シミュレータの前進 Euler と同じ向き。
    最後の時刻は nan)、学習器はサロゲートの右辺がその時刻の潜在と状態に返す dz/dt。
    右辺に渡す状態は置換系の kernel と同じく、群のゲートを潜在から作り直し、吸収した
    ゲートを V での定常値にし、残りは元のモデルの値。
    膜電位も同定する form=sindy は右辺が流入電流を読むので、ここでは扱わない (None)。
    """
    config, cell = s.config, s.config.cell_type
    if config.form != "hybrid":
        return None
    x = gate_matrix(ds, comp_id)
    v = potential(ds, comp_id)
    dt = float(ds["time"][1] - ds["time"][0])
    pairs = list(zip(s.parts, config.groups, _offsets(config), strict=True))
    z = np.concatenate(
        [np.asarray(c.encode(x[:, config.positions(g)])) for (c, _), g, _ in pairs],
        axis=1,
    )
    blocks = [
        (g.gates, start, start + g.d, c.decode, _rhs_fn(r))
        for (c, r), g, start in pairs
    ]

    def one(zt, vt, xt):
        st = {name: xt[i] for i, name in enumerate(cell.states)}
        st |= absorbed(cell.name, config.absorbed, params, vt)
        for gates, a, b, decode, _ in blocks:
            values = decode(zt[a:b])
            for j, gate in enumerate(gates):
                st[gate] = values[j]
        return jnp.concatenate(
            [jnp.atleast_1d(rhs(zt[a:b], vt, st, params)) for _, a, b, _, rhs in blocks]
        )

    batch = jax.jit(jax.vmap(one))
    learned = np.concatenate(
        [
            np.asarray(
                batch(
                    z[i : i + _SLOPE_CHUNK],
                    v[i : i + _SLOPE_CHUNK],
                    x[i : i + _SLOPE_CHUNK],
                )
            )
            for i in range(0, len(v), _SLOPE_CHUNK)
        ],
        axis=0,
    )
    slope = np.full_like(z, np.nan)
    slope[:-1] = np.diff(z, axis=0) / dt
    return slope, learned


@dataclass(frozen=True)
class CompFit:
    """学習区画 1 つの、学習器の潜在 (時刻 × 潜在)、再構成の RMSE (時刻ごと)、
    潜在の傾きの正解 slope (前進差分) と学習器の右辺が返す dz/dt の rhs
    (どちらも時刻 × 潜在。form=sindy は None)、
    全区間のまとめ summary (再構成の MSE と、それをゲートの分散で割ったもの)。"""

    latent: np.ndarray
    recon_rmse: np.ndarray
    slope: np.ndarray | None
    rhs: np.ndarray | None
    summary: dict[str, float]


_CURVE = 201  # decode の曲線の点の数


def decode_curves(
    compressor: Any,
    latents: list[str],
    ranges: list[tuple[float, float]],
    center: np.ndarray,
) -> list[dict]:
    """潜在の軸ごとに、ほかの軸を center に置いて範囲 ranges を掃いた decode の曲線。"""
    axes = []
    for k, name in enumerate(latents):
        grid = np.linspace(*ranges[k], _CURVE)
        z = np.tile(np.asarray(center, dtype=float), (_CURVE, 1))
        z[:, k] = grid
        gates = np.asarray(compressor.decode(jnp.asarray(z)))
        axes.append({"latent": name, "z": grid.tolist(), "gates": gates.T.tolist()})
    return axes


def describe_group(
    config: Config, group: Group, part: tuple[Any, Any], data: xr.Dataset | None
) -> tuple[dict[str, dict], dict[int, CompFit]]:
    """学習器 1 つの、図が読むファイル (run の中のパス → 中身) と学習区画の CompFit。

    compressor/decode.json は潜在 → ゲートの曲線 (潜在の軸ごと) と区画の型の状態の
    並び、compressor/own.json と rhs/own.json は圧縮器と右辺に固有の中身 (PCA は向きと
    平均も、KAM の右辺は kind だけ)。学習しない KAM は学習データを読まず、ψ の範囲で
    曲線を描き、学習区画も CompFit も持たない。学習する学習器は、学習データの上の
    潜在の範囲と平均で曲線を描く。
    """
    compressor, rhs = part
    latent_names = group_latents(group)
    files: dict[str, dict] = {
        "compressor/own.json": compressor.describe(),
        "rhs/own.json": rhs.describe(),
    }

    def decode(ranges: list[tuple[float, float]], center: np.ndarray) -> dict:
        return {
            "gates": group.gates,
            "order": config.cell_type.states,
            "axes": decode_curves(compressor, latent_names, ranges, center),
        }

    if group.compressor == "kam":
        files["compressor/decode.json"] = decode([compressor.psi_range], np.zeros(1))
        return files, {}
    if data is None:
        raise ValueError(f"{group.compressor} の群は学習データの上で書き出す")
    net = config._net()
    ids = config.train_ids()
    gates = _group_gates(config, group, data)
    latents = [np.asarray(compressor.encode(g)) for g in gates]
    recons = [np.asarray(compressor.decode(jnp.asarray(z))) for z in latents]
    every = np.concatenate(latents, axis=0)
    low, high = every.min(0), every.max(0)
    ranges = [(float(lo), float(hi)) for lo, hi in zip(low, high, strict=True)]
    files["compressor/decode.json"] = decode(ranges, every.mean(0))
    alone = Surrogate(config, [part])
    fits = {}
    for pos, comp_id in enumerate(ids):
        slopes = latent_slopes(alone, data, comp_id, net.nodes[comp_id].resolved_params)
        mse = float(np.mean((gates[pos] - recons[pos]) ** 2))
        fits[comp_id] = CompFit(
            latent=latents[pos],
            recon_rmse=np.sqrt(np.mean((gates[pos] - recons[pos]) ** 2, axis=1)),
            slope=slopes[0] if slopes else None,
            rhs=slopes[1] if slopes else None,
            summary={
                "reconstruction_mse": mse,
                "reconstruction_mse_ratio": mse / float(np.var(gates[pos])),
            },
        )
    return files, fits
