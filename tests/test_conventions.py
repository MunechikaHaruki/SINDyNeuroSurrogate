"""研究コードの約束の機械検査: import の向き、基盤の道具を使わないこと、`_` の名前。"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent / "neurosurrogate"

# 置き場 → import してよい置き場。physics は元のモデルを解くだけで、学習を知らない。
# 直下 (spikes・run) は両方を使う。
_ALLOWED = {
    "physics": {"physics"},
    "surrogate": {"physics", "surrogate"},
    "": {"physics", "surrogate", ""},
}
# 計算は記録・設定・描画の基盤を知らない (保存は run.py の置き場、描画は evaluation/)。
_INFRA = {"mlflow", "hydra", "omegaconf", "matplotlib"}


def _files() -> dict[str, ast.Module]:
    """neurosurrogate/ からのパス (physics/cells など) → 構文木。"""
    trees = {}
    for path in sorted(ROOT.rglob("*.py")):
        key = path.relative_to(ROOT).with_suffix("").as_posix()
        trees[key] = ast.parse(path.read_text())
    return trees


def _place(key: str) -> str:
    """physics/cells → physics、run → 直下 ("")。"""
    return key.split("/")[0] if "/" in key else ""


def _imports(key: str, tree: ast.Module) -> list[tuple[str | None, str, list[str]]]:
    """(neurosurrogate の中なら置き場, import 元, 取り出した名前)。"""
    found: list[tuple[str | None, str, list[str]]] = []
    here = key.split("/")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = [a.name for a in node.names]
            module = node.module or ""
            if node.level == 0:
                inside = module.startswith("neurosurrogate.")
                target = module.removeprefix("neurosurrogate.") if inside else None
            else:
                base = here[: len(here) - (node.level - 1)]
                target = "/".join([*base, *module.split(".")])
            place = None if target is None else _place(target.replace(".", "/"))
            found.append((place, module, names))
        elif isinstance(node, ast.Import):
            found.extend((None, a.name, []) for a in node.names)
    return found


def test_every_file_is_in_a_known_place() -> None:
    assert {_place(k) for k in _files()} <= set(_ALLOWED)


def test_import_direction() -> None:
    for key, tree in _files().items():
        for place, module, _ in _imports(key, tree):
            if place is not None:
                assert place in _ALLOWED[_place(key)], f"{key} → {module}"


def test_no_infra_imports() -> None:
    for key, tree in _files().items():
        for _, module, _ in _imports(key, tree):
            assert module.split(".")[0] not in _INFRA, f"{key} が {module} を使う"


def test_private_names_stay_in_their_file() -> None:
    for key, tree in _files().items():
        for place, module, names in _imports(key, tree):
            private = [n for n in names if n.startswith("_")]
            assert place is None or not private, (
                f"{key} が {module} の {private} を使う"
            )
