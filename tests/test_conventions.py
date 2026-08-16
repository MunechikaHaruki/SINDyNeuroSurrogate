"""**公開範囲の綴りが実態と一致すること**の機械検査。

規約: module 直下の名前は、**他の module から参照されるものだけが公開** (`_` 無し)。
自分の module の中でしか使わないものは `_` を付ける — `_` が「外から呼ばれない」の
印として信用できる状態を保つ (import 元を grep しないと公開範囲が分からない、を防ぐ)。

判定は import 文と属性アクセスの静的解析。動的にしか呼ばれない入口 (Hydra の
`_target_`、marimo の app、コンソールスクリプト) は `_EXEMPT` で明示的に免除する
= **免除もコードに書いてあるものだけ**。
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
# 検査対象 (ドメイン層 + 入口)。tests/ は「消費者」側としてだけ読む。
SRC_DIRS = ("neurosurrogate", "scripts")
CONSUMER_DIRS = (*SRC_DIRS, "tests")

# 動的に解決される入口 = 静的解析からは参照が見えない。
_EXEMPT = {
    ("scripts/main.py", "main"),  # Hydra の @hydra.main エントリ
    ("scripts/marimo.py", "app"),  # marimo が module 属性として拾う
}
# **名前空間ごと**外へ渡す module = 公開範囲が「module 直下の全部」。
# `sindy._catalog` が `vars(hh) | vars(traub)` を lambdify の名前空間に注入するので、
# レート関数は import されなくても**名前が API** (yaml の library_specs が文字列で
# 引く)。
_EXEMPT_MODULES = {
    "neurosurrogate/neurons/compartments/hh.py",
    "neurosurrogate/neurons/compartments/traub.py",
}
# 名前を文字列で受け取る呼び出し (monkeypatch など) の引数も参照とみなす。
_BY_NAME = {"setattr", "getattr", "hasattr", "delattr"}


def _py_files(dirs: tuple[str, ...]) -> list[pathlib.Path]:
    return sorted(
        p
        for d in dirs
        for p in (ROOT / d).rglob("*.py")
        if "__pycache__" not in p.parts
    )


def _module_path(mod: str, base: pathlib.Path, level: int) -> pathlib.Path | None:
    """import 文の module 指定 → ファイル。相対 import は `base` から遡り、
    絶対 import は import root (repo 直下と `scripts/`。後者は marimo/Hydra の
    入口が sys.path に持つ) から引く。"""
    if level:
        pkg = base.parent
        for _ in range(level - 1):
            pkg = pkg.parent
        roots = [pkg]
    else:
        roots = [ROOT, ROOT / "scripts"]
    for root in roots:
        target = root
        for part in mod.split(".") if mod else []:
            target = target / part
        for cand in (target.with_suffix(".py"), target / "__init__.py"):
            if cand.exists():
                return cand
    return None


def _defined() -> dict[tuple[pathlib.Path, str], int]:
    """module 直下の公開名 → 行番号 (`__init__.py` は再 export の場なので除く)。"""
    out: dict[tuple[pathlib.Path, str], int] = {}
    for path in _py_files(SRC_DIRS):
        if path.name == "__init__.py":
            continue
        for node in ast.parse(path.read_text()).body:
            names = []
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            for name in names:
                if not name.startswith("_"):
                    out[(path, name)] = node.lineno
    return out


def _imported_elsewhere() -> set[tuple[pathlib.Path, str]]:
    """**他の module から**参照されている (定義元ファイル, 名前)。

    見るのは `from x import name` と、`import x.y` + `x.y.name` の属性アクセス。
    属性名は定義元を特定せず全候補に当てる (過検出側 = 規約を緩める方向にだけ外す)。
    """
    used: set[tuple[pathlib.Path, str]] = set()
    attrs: set[str] = set()
    for path in _py_files(CONSUMER_DIRS):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                src = _module_path(node.module or "", path, node.level)
                if src is None or src == path:
                    continue
                for alias in node.names:
                    used.add((src, alias.name))
                    # `from pkg import module` (module をまるごと) → 属性経由の使用
                    sub = _module_path(
                        f"{node.module}.{alias.name}" if node.module else alias.name,
                        path,
                        node.level,
                    )
                    if sub is not None and sub != path:
                        used.update((sub, a) for a in attrs)
            elif isinstance(node, ast.Attribute):
                attrs.add(node.attr)
            elif isinstance(node, ast.Call):
                attrs |= _names_passed_as_strings(node)
    # 属性アクセスは module を特定しないので、定義元すべてに当てる
    return used | {(path, name) for (path, name) in _defined() if name in attrs}


def _names_passed_as_strings(call: ast.Call) -> set[str]:
    """`setattr(mod, "NAME", ...)` 系の呼び出しが文字列で名指しする名前。
    monkeypatch はこの形しか取らないので、これを見ないと差し替え対象が「参照なし」に
    見えてしまう。"""
    name = (
        call.func.attr
        if isinstance(call.func, ast.Attribute)
        else getattr(call.func, "id", "")
    )
    if name not in _BY_NAME:
        return set()
    return {
        arg.value
        for arg in call.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }


def _yaml_strings() -> str:
    return "\n".join(p.read_text() for p in (ROOT / "scripts" / "conf").rglob("*.yaml"))


def test_public_names_are_actually_imported_elsewhere() -> None:
    """`_` の無い module 直下の名前は、必ずどこか別の module が参照している。

    落ちたら 2 択: **他所から呼ばないなら `_` を付ける**、呼ぶなら呼ぶ側を足す
    (どこからも呼ばれないなら消す)。"""
    used = _imported_elsewhere()
    yaml = _yaml_strings()
    bad = [
        f"{path.relative_to(ROOT)}:{line} {name}"
        for (path, name), line in sorted(_defined().items())
        if (path, name) not in used
        and (str(path.relative_to(ROOT)), name) not in _EXEMPT
        and str(path.relative_to(ROOT)) not in _EXEMPT_MODULES
        and name not in yaml
    ]
    assert not bad, (
        "他 module から参照されないのに公開の綴り (`_` を付ける):\n" + "\n".join(bad)
    )


def test_private_modules_are_not_imported_from_outside() -> None:
    """`_` 始まりの module は、そのパッケージの外から import しない
    (AGENTS.md の規約)。"""
    bad = []
    for path in _py_files(CONSUMER_DIRS):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            src = _module_path(node.module or "", path, node.level)
            if src is None or src.parent == path.parent:
                continue
            if src.name.startswith("_") and src.name != "__init__.py":
                bad.append(f"{path.relative_to(ROOT)}:{node.lineno} → {node.module}")
    assert not bad, "パッケージ外から `_` module を import:\n" + "\n".join(bad)
