"""SINDy の右辺: 項ライブラリの疎な線形結合を疎回帰で同定する。

項は sympy の式で書き、役割の記号 V (電位)・g (潜在)・u (流入電流) を、同定器へ渡す
列へ束縛する。項の並びは pysindy の feature の並びと一致させ、fit のときに照合する。
"""

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax.numpy as jnp
import numpy as np
import pysindy as ps
import sympy as sp
from sympy.core.function import AppliedUndef

from ..physics.cells import RATE_COSTS, RATES, OpCost

V, G, U = sp.symbols("V g u")  # 電位 / 潜在 / 流入電流

# --- 項のカタログ ----------------------------------------------------------------


@dataclass(frozen=True)
class _Term:
    """候補の項 1 つ。args は式に現れる役割の記号 (V, g, u の順)。"""

    expr: sp.Expr
    args: tuple[sp.Symbol, ...]
    func: Callable = field(compare=False)

    def bound(self, *cols: sp.Symbol) -> sp.Expr:
        """役割の記号を列の記号へ置き換えた式。feature の同一性はこの式。"""
        return self.expr.subs(dict(zip(self.args, cols, strict=True)))

    def name(self, *names: str) -> str:
        """pysindy が feature 名を問うときの答え。"""
        return str(self.bound(*map(sp.Symbol, names)))


def _term(expr: sp.Expr) -> _Term:
    args = tuple(s for s in (V, G, U) if s in expr.free_symbols)
    return _Term(expr, args, sp.lambdify(args, expr, modules=[RATES, "jax"]))


def _rate(name: str) -> sp.Expr:
    """レート関数は数値に安定な実装を壊さないよう、未定義関数のまま持つ。"""
    return sp.Function(name)(V)


# 定数項。記号を持たないので V に相乗りさせる (値は常に 1)。
_ONE = _Term(sp.Integer(1), (V,), sp.lambdify(V, sp.Integer(1), modules="jax"))


def _rate_terms(model: str, gates: str) -> dict[str, list[_Term]]:
    """1 モデルのレートの項。type どうしは互いに素で、組み合わせは preset が書く。"""
    pairs = [(f"alpha_{g}_{model}", f"beta_{g}_{model}") for g in gates]
    return {
        f"{model}_gate_forward": [_term(_rate(a)) for a, _ in pairs],
        f"{model}_gate_backward": [_term(_rate(b)) for _, b in pairs],
        f"{model}_gate_forward_product": [_term(_rate(a) * G) for a, _ in pairs],
        f"{model}_gate_backward_product": [_term(_rate(b) * G) for _, b in pairs],
        f"{model}_relaxation_decay": [
            _term((_rate(a) + _rate(b)) * G) for a, b in pairs
        ],
    }


CATALOG: dict[str, list[_Term]] = {
    **_rate_terms("hh", "mhn"),
    **_rate_terms("traub", "msncahrb"),
    "gate_poly_volt": [_term(G**p) for p in range(2, 5)]
    + [_term(G**p * V) for p in range(1, 5)],
    "basis": [_term(V), _term(U), _term(G), _ONE],
}


def _of_gates(terms: list[_Term], gates: str | None) -> list[_Term]:
    """レートを含む項を、ゲートが gates にあるもの (alpha_h_traub なら h) に絞る。"""
    if gates is None:
        return terms
    kept = []
    for t in terms:
        rate_gates = [f.func.__name__.split("_")[1] for f in t.expr.atoms(AppliedUndef)]
        if all(g in gates for g in rate_gates):
            kept.append(t)
    return kept


# --- 列への束縛 ------------------------------------------------------------------


@dataclass(frozen=True)
class Roles:
    """役割の記号 → 同定器へ渡す列の位置。"""

    V: int
    g: list[int]
    u: int | None = None

    def bindings(self, args: tuple[sp.Symbol, ...], latents: list[int] | None):
        """項を並べる本数ぶんの列の束縛。g を含む項は潜在ごとに 1 本。"""
        if U in args and self.u is None:
            return []
        if G not in args:
            return [self._bind(args, None)]
        ks = range(len(self.g)) if latents is None else latents
        return [self._bind(args, self.g[k]) for k in ks]

    def _bind(self, args, gcol):
        cols = {V: self.V, G: gcol, U: self.u}
        return [cols[a] for a in args]


@dataclass(frozen=True)
class _Bundle:
    """同じ列へ束縛される項のまとまり (pysindy の CustomLibrary 1 つ)。"""

    terms: list[_Term]
    inputs: list[int]

    def exprs(self, columns: list[sp.Symbol]) -> list[sp.Expr]:
        bound = [columns[i] for i in self.inputs]
        return [t.bound(*bound) for t in self.terms]


def _bundles(library_specs: list[dict], roles: Roles) -> list[_Bundle]:
    """library_specs ({type, latents?, gates?}) を列へ束縛した束の並び。"""
    bundles = []
    for spec in library_specs:
        if (t := spec["type"]) not in CATALOG:
            raise ValueError(f"未知 library type: {t!r}。対応 type: {sorted(CATALOG)}")
        by_args: defaultdict[tuple, list[_Term]] = defaultdict(list)
        for term in _of_gates(CATALOG[t], spec.get("gates")):
            by_args[term.args].append(term)
        for args, terms in by_args.items():
            for inputs in roles.bindings(args, spec.get("latents")):
                bundles.append(_Bundle(terms, inputs))
    return bundles


def op_cost(e: sp.Expr) -> OpCost:
    """式木の演算コスト。冪は mul(p−1)、積は mul(項数−1)、和は pm(項数−1)。"""
    own: OpCost
    if e.is_Symbol or e.is_Number:
        own = OpCost()
    elif isinstance(e, AppliedUndef):
        own = RATE_COSTS[e.func.__name__]
    elif isinstance(e, sp.Pow) and e.exp.is_Integer and int(e.exp) >= 1:
        own = OpCost(mul=int(e.exp) - 1)
    elif isinstance(e, sp.Mul):
        own = OpCost(mul=len(e.args) - 1)
    elif isinstance(e, sp.Add):
        own = OpCost(pm=len(e.args) - 1)
    else:
        raise ValueError(f"op_cost 未対応: {e!r}")
    return sum((op_cost(child) for child in e.args), own)


# --- 同定 ------------------------------------------------------------------------

_OPTIMIZERS: dict[str, type] = {"stlsq": ps.optimizers.STLSQ, "sr3": ps.optimizers.SR3}
_DEFAULT_OPTIMIZER = {
    "type": "stlsq",
    "threshold": 0.01,
    "normalize_columns": False,
    "alpha": 2.0,
}


@dataclass
class SINDy:
    """dx/dt = Ξ Θ(列)。targets は x の列の記号、inputs は u の列の記号。"""

    xi: np.ndarray
    targets: list[sp.Symbol]
    inputs: list[sp.Symbol]
    library_specs: list[dict]
    roles: Roles

    @property
    def columns(self) -> list[sp.Symbol]:
        return self.targets + self.inputs

    @cached_property
    def _bundles(self) -> list[_Bundle]:
        # lambdify した関数は pickle できないので保存せず、読んだあとに組み直す
        return _bundles(self.library_specs, self.roles)

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "_bundles"}

    @property
    def feature_exprs(self) -> list[sp.Expr]:
        """Ξ の列に対応する式 (並び = pysindy の feature)。重複は library の誤り。"""
        exprs = [e for b in self._bundles for e in b.exprs(self.columns)]
        if dup := [e for e, n in Counter(exprs).items() if n > 1]:
            raise ValueError(f"library間で feature 重複: {sorted(map(str, dup))}")
        return exprs

    def theta(self) -> Callable:
        """(列の値...) → Θ。"""
        bundles = self._bundles

        def theta(*cols):
            values = []
            for b in bundles:
                bound = [cols[i] for i in b.inputs]
                for t in b.terms:
                    values.append(t.func(*bound))
            return jnp.array(values, dtype=jnp.float64)

        return theta

    def metrics(self) -> dict[str, float]:
        nnz = int((self.xi != 0).sum())
        return {"nnz": nnz, "nnz_ratio": nnz / self.xi.size}

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "sindy",
            "targets": [sp.latex(t) for t in self.targets],
            "features": [sp.latex(e) for e in self.feature_exprs],
            "xi": np.asarray(self.xi).tolist(),
        }

    def opcost(self) -> OpCost:
        """Ξ の積和と、0 でない係数を持つ項の評価。"""
        nnz = np.count_nonzero(self.xi).item()
        active = np.any(self.xi != 0, axis=0)
        cost = OpCost(mul=nnz, pm=max(0, nnz - int(self.xi.shape[0])))
        for expr, on in zip(self.feature_exprs, active, strict=True):
            if on:
                cost = cost + op_cost(expr)
        return cost


def fit_sindy(
    x_names: list[str],
    u_names: list[str],
    x: list[np.ndarray],
    u: list[np.ndarray],
    t: np.ndarray,
    roles: Roles,
    *,
    library_specs: list[dict],
    optimizer: dict = _DEFAULT_OPTIMIZER,
) -> SINDy:
    """区画ごとの軌道 (x, u) から Ξ を疎回帰で同定する。軌道をまたいで微分しない。"""
    model = SINDy(
        xi=np.empty(0),
        targets=[sp.Symbol(n) for n in x_names],
        inputs=[sp.Symbol(n) for n in u_names],
        library_specs=library_specs,
        roles=roles,
    )
    library = ps.GeneralizedLibrary(
        [
            ps.CustomLibrary(
                library_functions=[t.func for t in b.terms],
                function_names=[t.name for t in b.terms],
            )
            for b in model._bundles
        ],
        inputs_per_library=[b.inputs for b in model._bundles],
    )
    opt = {k: v for k, v in optimizer.items() if v is not None}
    sindy = ps.SINDy(
        feature_library=library, optimizer=_OPTIMIZERS[opt.pop("type")](**opt)
    )
    sindy.fit(x, u=u, t=[t] * len(x), feature_names=[str(s) for s in model.columns])
    model.xi = sindy.coefficients()
    if (names := sindy.get_feature_names()) != [str(e) for e in model.feature_exprs]:
        raise ValueError(f"pysindy の feature 順が展開結果と不一致: {names}")
    return model
