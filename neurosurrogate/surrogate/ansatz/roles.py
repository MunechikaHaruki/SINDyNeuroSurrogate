from dataclasses import dataclass

import sympy as sp

# ---------------------------------------------------------------------------
# 方程式の列構造 = ansatz のドメイン。役割記号 V/g/u の定義はここが唯一の源で、
# 項カタログ (sindy/catalog.py) は式をこの記号で書き、Roles が ansatz ごとの列
# index へ束縛する。SINDy 入力行列の各列が何か (電位/ゲート/外部電流) は ansatz
# ごとに順序が違うため、各 ansatz が n_components から fit 内で構築する。
# ---------------------------------------------------------------------------

V, G, U = sp.symbols("V g u")  # 電位 / ゲート(=隠れ変数) / 外部電流


@dataclass(frozen=True)
class Roles:
    """役割記号 → 列 index。V=電位(単一列) / g=ゲート列群 (=隠れ変数) / u=外部電流
    (任意)。yaml の手書き index を廃し、項の式に現れる記号でここへ束縛する。yaml が
    指定できる番号は g の序数のみ (V/u は各項が固定保有)。"""

    V: int
    g: list[int]
    u: int | None = None

    def bindings(
        self, args: tuple[sp.Symbol, ...], latents: list[int] | None = None
    ) -> list[list[int]]:
        """項の引数記号を実列へ束縛。返す本数 = その項を何本並べるかで、args が決める。
        latents は展開先 latent の序数 (None=全 latent)。"""
        if U in args and self.u is None:
            return []  # u を要求する項は u の無い ansatz では脱落
        if G not in args:
            return [self._bind(args, None)]  # latent 非依存 → 1 本
        ks = range(len(self.g)) if latents is None else latents
        return [self._bind(args, self.g[k]) for k in ks]  # latent ごとに 1 本

    def _bind(self, args: tuple[sp.Symbol, ...], gcol: int | None) -> list[int]:
        """1 本分の列束縛。gcol = この本が使う g 列 (g を持たない項では None)。"""
        cols = {V: self.V, G: gcol, U: self.u}
        return [cols[a] for a in args]  # type: ignore[misc]
