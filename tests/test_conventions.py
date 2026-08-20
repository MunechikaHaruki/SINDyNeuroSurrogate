"""**設計規約が実態と一致すること**の機械検査。

検査するのは 3 系統:

1. **公開範囲の綴り** — module 直下の名前も module 名も、**他から参照されるものだけ**
   が `_` 無し。内側でしか使わないものは `_` を付ける = `_` が「外から呼ばれない」の
   印として信用できる状態を保つ (import 元を grep しないと公開範囲が分からない、を防ぐ)
2. **依存の向き** — `neurosurrogate/` 内の import が `_LAYERS` の許可を超えない
   (`docs/architecture.md` の層の宣言をそのまま実行可能にしたもの)
3. **ドメイン層の独立** — `neurosurrogate/` が marimo/MLflow/Hydra を import しない

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
    "neurosurrogate/neurons/hh.py",
    "neurosurrogate/neurons/traub.py",
}
# 名前を文字列で受け取る呼び出し (monkeypatch など) の引数も参照とみなす。
_BY_NAME = {"setattr", "getattr", "hasattr", "delattr"}

# `_` 無しの module 名でも、外から import されない入口 (Hydra/marimo が直接起動する)。
_EXEMPT_ENTRY_MODULES = {"scripts/main.py", "scripts/marimo.py"}

# --- 依存の向き (docs/architecture.md の宣言) ---------------------------------
# 場所 → 層 (module path の前方一致。長い方が勝つ = ファイル単位で層を割れる)。
# 新しいディレクトリを足したらここに書く必要がある = **層の所属は必ず宣言される**。
_GROUP_OF = {
    "neurosurrogate/__init__": "base",  # package import 時に jax_enable_x64 を張る入口
    "neurosurrogate/core/": "base",
    "neurosurrogate/artifact/model": "base",
    "neurosurrogate/artifact/plotting": "base",
    "neurosurrogate/artifact/bundle": "bundle",
    "neurosurrogate/neurons/": "neurons",
    "neurosurrogate/sim/run": "sim_exec",
    "neurosurrogate/sim/artifacts": "sim_exec",
    "neurosurrogate/sim/": "sim_desc",
    "neurosurrogate/surrogate/": "surrogate",
}
# 層 → import してよい層 (自分自身は常に可)。推移的な許可も**明示的に**書く
# = 「どこから何が見えるか」がこの表だけで読める。
_LAYERS = {
    "base": frozenset(),  # 他ディレクトリを一切 import しない基盤
    "neurons": frozenset({"base"}),
    "sim_desc": frozenset({"base", "neurons"}),
    "surrogate": frozenset({"base", "neurons", "sim_desc"}),
    "sim_exec": frozenset({"base", "neurons", "sim_desc", "surrogate"}),
    # 合流点。全部見てよいのはここだけ
    "bundle": frozenset({"base", "neurons", "sim_desc", "surrogate", "sim_exec"}),
}

# ドメイン層が触ってはいけない基盤 (実行/記録の入口は scripts/ だけが知る)。
_INFRA_ROOTS = {"marimo", "mlflow", "hydra", "omegaconf"}


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


def _import_nodes(path: pathlib.Path) -> list[ast.Import | ast.ImportFrom]:
    """ファイル中の import 文だけ (行番号を持つ型に絞ってから回す)。"""
    return [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]


def _dotted_targets(node: ast.Import | ast.ImportFrom, base: pathlib.Path) -> list[str]:
    """import 文が名指す module (相対は絶対へ直す)。`from pkg import mod` の mod も
    候補に足す — 層の判定に `pkg` だけでは足りない (`sim` は spec と run で層が違う)。
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        pkg = base.parent
        for _ in range(node.level - 1):
            pkg = pkg.parent
        prefix = str(pkg.relative_to(ROOT)).replace("/", ".")
        mod = f"{prefix}.{node.module}" if node.module else prefix
    else:
        mod = node.module or ""
    return [mod, *(f"{mod}.{alias.name}" for alias in node.names)]


def _imported_modules(
    node: ast.Import | ast.ImportFrom, base: pathlib.Path
) -> list[pathlib.Path]:
    """import 文が名指す module のファイル。`from pkg import mod` の mod と
    `import pkg.mod` の両形式を見る (後者だけで参照される module がある)。"""
    if isinstance(node, ast.Import):
        return [
            src
            for alias in node.names
            if (src := _module_path(alias.name, base, 0)) is not None
        ]
    names = [node.module or "", *(f"{node.module}.{a.name}" for a in node.names)]
    return [
        src
        for name in names
        if (src := _module_path(name, base, node.level)) is not None
    ]


def _group_of(module_path: str) -> str | None:
    """module path (拡張子なし) → 層。前方一致が複数当たったら長い方が勝つ。"""
    hit = [
        (prefix, g)
        for prefix, g in _GROUP_OF.items()
        if f"{module_path}/".startswith(prefix)
    ]
    return max(hit, key=lambda kv: len(kv[0]))[1] if hit else None


def _defined() -> dict[tuple[pathlib.Path, str], int]:
    """module 直下の公開名 → 行番号。"""
    out: dict[tuple[pathlib.Path, str], int] = {}
    for path in _py_files(SRC_DIRS):
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


def test_public_modules_are_imported_from_outside() -> None:
    """`_` の無い module は、必ずそのディレクトリの外から import されている。

    名前に対する `test_public_names_are_actually_imported_elsewhere` と同じ規約を
    module 名に当てる。落ちたら 2 択: **外から使わないなら `_`**、使うなら使う側を足す。
    """
    outside: set[pathlib.Path] = set()
    for path in _py_files(CONSUMER_DIRS):
        for node in _import_nodes(path):
            outside |= {
                src
                for src in _imported_modules(node, path)
                if src.parent != path.parent
            }
    yaml = _yaml_strings()
    bad = [
        str(path.relative_to(ROOT))
        for path in _py_files(SRC_DIRS)
        if not path.name.startswith("_")
        and path not in outside
        and str(path.relative_to(ROOT)) not in _EXEMPT_ENTRY_MODULES
        and path.stem not in yaml
    ]
    assert not bad, "外から import されない module が公開の綴り:\n" + "\n".join(bad)


def test_import_direction_follows_layers() -> None:
    """`neurosurrogate/` 内の import が `_LAYERS` の許可を超えない (依存の向き)。

    落ちたら、まず**呼ぶ側の層を疑う** (下位層が上位層を必要とするのは責務の置き場所が
    間違っているサイン)。層の定義そのものを変えるのは設計判断なので、独断で `_LAYERS` を
    書き換えない。"""
    bad = set()
    for path in _py_files(("neurosurrogate",)):
        src_group = _group_of(str(path.relative_to(ROOT).with_suffix("")))
        if src_group is None:
            bad.add(f"{path.relative_to(ROOT)} の層が `_GROUP_OF` に無い")
            continue
        for node in _import_nodes(path):
            for mod in _dotted_targets(node, path):
                # `from pkg import mod` は pkg と pkg.mod の両方が候補 = 同じ 1 行が
                # 2 度出る。層だけ見て潰す (どちらが当たっても違反の中身は同じ)。
                dst_group = _group_of(mod.replace(".", "/"))
                if dst_group not in (None, src_group) and (
                    dst_group not in _LAYERS[src_group]
                ):
                    bad.add(
                        f"{path.relative_to(ROOT)}:{node.lineno} "
                        f"{src_group} → {dst_group}"
                    )
    assert not bad, "依存の向きに反する import:\n" + "\n".join(sorted(bad))


def test_domain_layer_does_not_import_infra() -> None:
    """`neurosurrogate/` は marimo/MLflow/Hydra を知らない (入口は `scripts/`)。"""
    bad = []
    for path in _py_files(("neurosurrogate",)):
        for node in _import_nodes(path):
            for mod in _dotted_targets(node, path):
                if mod.split(".")[0] in _INFRA_ROOTS:
                    bad.append(f"{path.relative_to(ROOT)}:{node.lineno} {mod}")
    assert not bad, "ドメイン層からの基盤 import:\n" + "\n".join(bad)


def test_no_dunder_all() -> None:
    """`__all__` は置かない (公開範囲は綴りが持つ = 二重管理にしない)。"""
    bad = [
        f"{path.relative_to(ROOT)}:{node.lineno}"
        for path in _py_files(SRC_DIRS)
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    ]
    assert not bad, "`__all__` は定義しない (AGENTS.md):\n" + "\n".join(bad)


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
