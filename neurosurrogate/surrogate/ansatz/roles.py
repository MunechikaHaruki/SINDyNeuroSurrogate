from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 方程式の列構造 = ansatz のドメイン。SINDy 入力行列の各列が何か (電位/ゲート/
# 外部電流) は ansatz ごとに順序が違う。Roles は「役割名 → 列 index」を持ち、
# ライブラリ項の args シンボル名 (V/g) を列へ束縛する翻訳器。各 ansatz が
# n_components から `_roles` property で宣言的に構築する。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Roles:
    """入力列の役割 → 列 index。V=電位(単一列) / g=ゲート列群 / u=外部電流(任意)。
    yaml の手書き index を廃し、項の args シンボル名でここへ束縛する。"""

    V: int
    g: list[int]
    u: int | None = None

    def bind(self, argnames: list[str], gcol: int | None) -> list[int]:
        """項の args シンボル名を列 index に束縛。V→自列、g→gcol (展開時の1列)。"""
        return [self.V if n == "V" else gcol for n in argnames]  # type: ignore[misc]

    def basis_cols(self, role_names: list[str] | None, n_inputs: int) -> list[int]:
        """basis の対象列。role_names 未指定なら全入力列、指定なら該当役割列を昇順。"""
        if role_names is None:
            return list(range(n_inputs))
        cols: set[int] = set()
        for name in role_names:
            value = getattr(self, name)
            cols.update(value if isinstance(value, list) else [value])
        return sorted(cols)
