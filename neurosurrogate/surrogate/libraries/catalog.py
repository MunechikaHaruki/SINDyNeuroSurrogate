from collections.abc import Callable

import sympy as sp

from ...compartments.hh import (
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
)
from .entry import LibraryEntry

# ---------------------------------------------------------------------------
# カタログ層。項カタログ (式リスト) が唯一の真実源。yaml spec が指定できるのは
# type と latent (隠れ変数) 序数のみで、V/u/定数を保有するか g を latent へ展開
# するかは、各項の式に現れる記号がここで決めている (_entry の args 自動導出)。
# ---------------------------------------------------------------------------

V, G, U = sp.symbols("V g u")  # 電位 / ゲート(=隠れ変数) / 外部電流

_RATE_IMPL: dict[str, Callable] = {
    "alpha_m": alpha_m,
    "beta_m": beta_m,
    "alpha_h": alpha_h,
    "beta_h": beta_h,
    "alpha_n": alpha_n,
    "beta_n": beta_n,
}
# レート式 alpha_m(V) 等。HH レート関数は数値安定な exp 実装を壊さないため未定義
# 関数のまま構造だけ持ち、lambdify 時に _RATE_IMPL を注入する。
_R: dict[str, sp.Expr] = {nm: sp.Function(nm)(V) for nm in _RATE_IMPL}


def _entry(expr: sp.Expr, *args: sp.Symbol) -> LibraryEntry:
    """式 1 つ → 候補項。args (arity 兼 束縛順) は式に現れる記号から V,g,u の順で
    自動導出する。定数項のみ arity 0 を避けるため相乗り先の記号を明示する。"""
    args = args or tuple(s for s in (V, G, U) if s in expr.free_symbols)
    return LibraryEntry(
        expr=expr,
        args=args,
        func=sp.lambdify(args, expr, modules=[_RATE_IMPL, "jax"]),
    )


_GATE_RATES = ["alpha_m", "beta_m", "alpha_h", "beta_h", "alpha_n", "beta_n"]
_FORWARD_RATES = ["alpha_m", "alpha_h", "alpha_n"]
_RELAX_PAIRS = [("alpha_m", "beta_m"), ("alpha_h", "beta_h"), ("alpha_n", "beta_n")]

LIB_ENTRIES: dict[str, list[LibraryEntry]] = {
    "hh_gate": [_entry(_R[nm]) for nm in _GATE_RATES],
    "hh_gate_product": [_entry(_R[nm] * G) for nm in _GATE_RATES],
    "hh_gate_forward": [_entry(_R[nm]) for nm in _FORWARD_RATES],
    "hh_gate_forward_product": [_entry(_R[nm] * G) for nm in _FORWARD_RATES],
    "hh_relaxation_driver": [_entry(_R[a]) for a, _ in _RELAX_PAIRS],
    "hh_relaxation_decay": [_entry((_R[a] + _R[b]) * G) for a, b in _RELAX_PAIRS],
    "gate_poly_volt": [_entry(e) for p in range(1, 5) for e in (G**p, G**p * V)],
    # 射影 + 定数。V/u 射影は latent 非依存の 1 束 (u 射影は u の無い ansatz では
    # expand が落とす)、latent 射影は spec["latents"] で選択展開される。定数 1 は
    # arity 0 を避けるため V に相乗りさせる。
    "basis": [_entry(V), _entry(U), _entry(G), _entry(sp.Integer(1), V)],
}
