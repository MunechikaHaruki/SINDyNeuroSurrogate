"""Kepler–Abbott–Marder の縮約。群のゲートを 1 本の等価電位 ψ にまとめる。

学習せず、元の区画の式から組む。群のゲートは定常値 x̄_i(ψ) で作り直し、ψ は
dψ/dt = Σ_i D_i ẋ_i / Σ_j D_j x̄_j'(ψ)、D_i = ∂(dV/dt)/∂x_i で動かす (Kepler 1992 の
式(6)の重みで ẋ_i を等価電位の速さへ写したもの)。φ 群 (膜電位への吸収) は扱わない。
サロゲートの群の圧縮器と右辺 (kam) として使い、学習した圧縮と同じ段取りで比べる。
導き方は context-router の冊 52.01、対話的に動かす版は evaluation/traub/traub.ts。
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .cells import CELLS, OpCost, total

# 等価電位を探す範囲 [mV]。x̄ はこの中で単調 (TS の equivalentRange と同じ)。
_RANGE = (-120.0, 80.0)
_RANGES = {"R": (-60.0, 80.0), "C": (-120.0, -10.0)}
_BISECT = 40


def _steady(cell: str, gate: str, p):
    """V ↦ x̄(V) = α/(α+β)。ゲートの式は x の 1 次式なので、x = 0, 1 の値から読む。"""
    deriv = CELLS[cell].derivs[gate]  # type: ignore[index]

    def steady(v):
        opening = deriv(p, v, {gate: jnp.zeros_like(v)})
        closing = deriv(p, v, {gate: jnp.ones_like(v)})
        return opening / (opening - closing)

    return steady


def _equivalent(steady, x: np.ndarray, low: float, high: float) -> np.ndarray:
    """等価電位 v = x̄⁻¹(x) を二分法で。値域の外の x は端に寄る。"""
    lo = np.full_like(x, low)
    hi = np.full_like(x, high)
    rising = float(steady(jnp.asarray(high))) > float(steady(jnp.asarray(low)))
    for _ in range(_BISECT):
        mid = (lo + hi) / 2
        below = (np.asarray(steady(jnp.asarray(mid))) < x) == rising
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return np.asarray((lo + hi) / 2)


class KamCompressor:
    """群のゲート ↔ ψ。encode は各ゲートの等価電位の平均 (射影と初期値に使う)。"""

    gate_inits: list[float]
    reconstruction: dict[str, float]

    def __init__(self, cell: str, gates: list[str], params: Any) -> None:
        self.cell = cell
        self.gates = gates
        self.params = params  # レートは区画の定数によらないので、学習区画のもの

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        potentials = []
        for j, gate in enumerate(self.gates):
            low, high = _RANGES.get(gate, _RANGE)
            steady = _steady(self.cell, gate, self.params)
            potentials.append(_equivalent(steady, x[:, j], low, high))
        return np.mean(potentials, axis=0)[:, None]

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        psi = z[..., 0]
        steadies = [_steady(self.cell, g, self.params)(psi) for g in self.gates]
        return jnp.stack(steadies, axis=-1)

    def metrics(self) -> dict[str, float]:
        return {f"kam/{k}": v for k, v in self.reconstruction.items()}

    def describe(self) -> dict[str, Any]:
        return {"kind": "kam", "gates": self.gates}

    def opcost(self) -> OpCost:
        """decode: ゲートごとに ψ でのレートの組と x̄ の割り算。"""
        costs = CELLS[self.cell].costs
        assert costs is not None
        return total([costs[g] + OpCost(pm=1, div=1) for g in self.gates])


class KamRhs:
    """dψ/dt = Σ_i D_i ẋ_i / Σ_j D_j x̄_j'(ψ)。ẋ_i は作り直したゲートでの元の式。"""

    def __init__(self, cell: str, gates: list[str]) -> None:
        self.cell = cell
        self.gates = gates

    def rhs(self):
        """(ψ, V, 状態の名前 → 値, params) → dψ/dt。"""
        cell = CELLS[self.cell]
        assert cell.dv is not None and cell.derivs is not None
        dv, derivs, gates, name = cell.dv, cell.derivs, self.gates, self.cell

        def dpsi(z, v, s, p):
            psi = z[0]
            flow = 0.0
            weight = 0.0
            for gate in gates:

                def current(x, gate=gate):
                    return dv(p, 0.0, v, {**s, gate: x})

                d = jax.grad(current)(s[gate])
                slope = jax.grad(_steady(name, gate, p))(psi)
                flow = flow + d * derivs[gate](p, v, s)
                weight = weight + d * slope
            return jnp.atleast_1d(flow / weight)

        return dpsi

    def metrics(self) -> dict[str, float]:
        return {}

    def describe(self) -> dict[str, Any]:
        return {"kind": "kam", "gates": self.gates}

    def opcost(self) -> OpCost:
        """ゲートごとに V での元の式、ψ での x̄' (レート 2 組分)、D_i と積和。目安。"""
        costs = CELLS[self.cell].costs
        assert costs is not None
        per_gate = [costs[g] * 3 + OpCost(pm=2, mul=4) for g in self.gates]
        return total(per_gate) + OpCost(div=1)
