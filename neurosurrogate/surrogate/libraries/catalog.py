import sympy as sp

from ...compartments import hh
from .entry import LibraryEntry

# ---------------------------------------------------------------------------
# カタログ層。項カタログ (式リスト) が唯一の真実源。yaml spec が指定できるのは
# type と latent (隠れ変数) 序数のみで、V/u/定数を保有するか g を latent へ展開
# するかは、各項の式に現れる記号がここで決めている (_entry の args 自動導出)。
# ---------------------------------------------------------------------------

V, G, U = sp.symbols("V g u")  # 電位 / ゲート(=隠れ変数) / 外部電流


def _R(name: str) -> sp.Expr:
    """レート式 alpha_m(V) 等。HH レート関数は数値安定な exp 実装を壊さないため
    未定義関数のまま構造だけ持ち、lambdify 時に hh の実装を注入する。"""
    return sp.Function(name)(V)


def _entry(expr: sp.Expr) -> LibraryEntry:
    """式 1 つ → 候補項。args (arity 兼 束縛順) は式に現れる記号から V,g,u の順で
    自動導出する。"""
    args = tuple(s for s in (V, G, U) if s in expr.free_symbols)
    return LibraryEntry(
        expr=expr, args=args, func=sp.lambdify(args, expr, modules=[vars(hh), "jax"])
    )


# 定数項。式に記号が無く args 自動導出では arity 0 になるため、V に相乗りさせる
# (値は引数を無視して常に 1)。
_ONE = LibraryEntry(
    expr=sp.Integer(1), args=(V,), func=sp.lambdify(V, sp.Integer(1), modules="jax")
)


_GATE_RATES = ["alpha_m", "beta_m", "alpha_h", "beta_h", "alpha_n", "beta_n"]
_FORWARD_RATES = ["alpha_m", "alpha_h", "alpha_n"]
_RELAX_PAIRS = [("alpha_m", "beta_m"), ("alpha_h", "beta_h"), ("alpha_n", "beta_n")]

LIB_ENTRIES: dict[str, list[LibraryEntry]] = {
    "hh_gate": [_entry(_R(nm)) for nm in _GATE_RATES],
    "hh_gate_product": [_entry(_R(nm) * G) for nm in _GATE_RATES],
    "hh_gate_forward": [_entry(_R(nm)) for nm in _FORWARD_RATES],
    "hh_gate_forward_product": [_entry(_R(nm) * G) for nm in _FORWARD_RATES],
    "hh_relaxation_driver": [_entry(_R(a)) for a, _ in _RELAX_PAIRS],
    "hh_relaxation_decay": [_entry((_R(a) + _R(b)) * G) for a, b in _RELAX_PAIRS],
    "gate_poly_volt": [_entry(e) for p in range(1, 5) for e in (G**p, G**p * V)],
    # 射影 + 定数。V/u 射影は latent 非依存の 1 束 (u 射影は u の無い ansatz では
    # expand が落とす)、latent 射影は spec["latents"] で選択展開される。
    "basis": [_entry(V), _entry(U), _entry(G), _ONE],
}
