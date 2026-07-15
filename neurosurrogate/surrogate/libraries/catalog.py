import sympy as sp

from ...compartments import hh, traub
from .entry import LibraryEntry

_RATE_NS = vars(hh) | vars(traub)  # lambdify にレート実装を注入する名前空間

# ---------------------------------------------------------------------------
# カタログ層。項カタログ (式リスト) が唯一の真実源。yaml spec が指定できるのは
# type と latent (隠れ変数) 序数のみで、V/u/定数を保有するか g を latent へ展開
# するかは、各項の式に現れる記号がここで決めている (_entry の args 自動導出)。
# ---------------------------------------------------------------------------

V, G, U = sp.symbols("V g u")  # 電位 / ゲート(=隠れ変数) / 外部電流


def _R(name: str) -> sp.Expr:
    """レート式 alpha_m(V) 等。レート関数は数値安定な exp 実装を壊さないため未定義
    関数のまま構造だけ持ち、lambdify 時に compartments の実装を注入する。"""
    return sp.Function(name)(V)


def _entry(expr: sp.Expr) -> LibraryEntry:
    """式 1 つ → 候補項。args (arity 兼 束縛順) は式に現れる記号から V,g,u の順で
    自動導出する。"""
    args = tuple(s for s in (V, G, U) if s in expr.free_symbols)
    return LibraryEntry(
        expr=expr, args=args, func=sp.lambdify(args, expr, modules=[_RATE_NS, "jax"])
    )


# 定数項。式に記号が無く args 自動導出では arity 0 になるため、V に相乗りさせる
# (値は引数を無視して常に 1)。
_ONE = LibraryEntry(
    expr=sp.Integer(1), args=(V,), func=sp.lambdify(V, sp.Integer(1), modules="jax")
)


# (alpha, beta) ペア列がモデル毎の唯一の入力。gate/forward/relaxation の各 type は
# ここから派生する。HH=3ゲート、Traub=V 依存 8 ゲート (Q は XI 依存のため対象外)。
_HH_PAIRS = [("alpha_m", "beta_m"), ("alpha_h", "beta_h"), ("alpha_n", "beta_n")]
_TRAUB_PAIRS = [(f"traub_alpha_{x}", f"traub_beta_{x}") for x in "msncahrb"]


def _rate_types(
    prefix: str, pairs: list[tuple[str, str]]
) -> dict[str, list[LibraryEntry]]:
    """1 モデルのレート項 type 一式 ((alpha,beta) ペア列から派生)。type 同士は
    互いに素 (同じ式を2つの type が持たない) で、合成は yaml 側の責務。
    例: forward + backward = 全レート、forward + relaxation_decay = 緩和形。"""
    return {
        # 順方向レート α(V)。緩和形 dg/dt = α - (α+β)·g の駆動項でもある
        f"{prefix}_gate_forward": [_entry(_R(a)) for a, _ in pairs],
        # 逆方向レート β(V)
        f"{prefix}_gate_backward": [_entry(_R(b)) for _, b in pairs],
        # α(V)·g
        f"{prefix}_gate_forward_product": [_entry(_R(a) * G) for a, _ in pairs],
        # β(V)·g。forward_product と組で dg/dt = α(1-g) - βg のゲート依存側を張る
        f"{prefix}_gate_backward_product": [_entry(_R(b) * G) for _, b in pairs],
        # 緩和形の減衰項。α·g と β·g を束ね、係数 1 個で真の構造に一致させる
        f"{prefix}_relaxation_decay": [_entry((_R(a) + _R(b)) * G) for a, b in pairs],
    }


LIB_ENTRIES: dict[str, list[LibraryEntry]] = {
    **_rate_types("hh", _HH_PAIRS),
    **_rate_types("traub", _TRAUB_PAIRS),
    # latent の高次項と V 積。素の g (1次) は latent 射影として basis が持つ。
    "gate_poly_volt": [_entry(G**p) for p in range(2, 5)]
    + [_entry(G**p * V) for p in range(1, 5)],
    # 射影 + 定数。V/u 射影は latent 非依存の 1 束 (u 射影は u の無い ansatz では
    # expand が落とす)、latent 射影は spec["latents"] で選択展開される。
    "basis": [_entry(V), _entry(U), _entry(G), _ONE],
}
