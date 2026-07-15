from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 方程式の列構造 = ansatz のドメイン。SINDy 入力行列の各列が何か (電位/ゲート/
# 外部電流) は ansatz ごとに順序が違う。Roles は「役割名 → 列 index」を持ち、
# ライブラリ項の args シンボル名 (V/g/u) を列へ束縛する翻訳器。各 ansatz が
# n_components から fit 内で構築する。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Roles:
    """入力列の役割 → 列 index。V=電位(単一列) / g=ゲート列群 (=隠れ変数) /
    u=外部電流(任意)。yaml の手書き index を廃し、項の args シンボル名でここへ
    束縛する。yaml が指定できる番号は g の序数のみ (V/u は各項が固定保有)。"""

    V: int
    g: list[int]
    u: int | None = None

    def bind(self, argnames: tuple[str, ...], gcol: int | None) -> list[int]:
        """項の args シンボル名を列 index に束縛。V/u→自列、g→gcol (展開時の1列)。
        u 群は u を持つ ansatz でのみ呼ばれる (expand が事前に落とす)。"""
        cols = {"V": self.V, "u": self.u, "g": gcol}
        return [cols[n] for n in argnames]  # type: ignore[misc]
