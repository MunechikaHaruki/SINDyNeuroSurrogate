"""neurosurrogate/FILES.md を docstring と import から作り、置いてあるものと照らす。

`uv run python tests/test_design.py` で作り直す (context-router の hook が呼ぶ)。
"""

import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT = REPO / "neurosurrogate"
FILES = ROOT / "FILES.md"
PLACES = ["physics", "surrogate", ""]


def _modules() -> dict[str, ast.Module]:
    """neurosurrogate/ からのパス (physics/cells など) → 構文木。__init__ は除く。"""
    trees = {}
    for path in sorted(ROOT.rglob("*.py")):
        if path.stem == "__init__":
            continue
        key = path.relative_to(ROOT).with_suffix("").as_posix()
        trees[key] = ast.parse(path.read_text())
    return trees


def _place(key: str) -> str:
    """physics/cells → physics、run → 直下 ("")。"""
    return key.split("/")[0] if "/" in key else ""


def _imported(key: str, tree: ast.Module, known: set[str]) -> set[str]:
    """key が import する neurosurrogate の中のモジュール。`from . import x` も拾う。"""
    found = set()
    here = key.split("/")[:-1]
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if node.level == 0:
            if not module.startswith("neurosurrogate"):
                continue
            target = module.removeprefix("neurosurrogate").lstrip(".")
            target = target.replace(".", "/")
        else:
            base = here[: len(here) - (node.level - 1)]
            target = "/".join([*base, *module.split(".")]) if module else "/".join(base)
        candidates = [target]
        for alias in node.names:
            candidates.append(f"{target}/{alias.name}".strip("/"))
        for candidate in candidates:
            if candidate in known and candidate != key:
                found.add(candidate)
    return found


def _reduced(graph: dict[str, set[str]]) -> list[tuple[str, str]]:
    """推移的な辺を除いた辺。ほかの道で届く辺は描かない。"""

    def reachable(start: str, skip: tuple[str, str]) -> set[str]:
        seen, stack = set(), [start]
        while stack:
            node = stack.pop()
            for nxt in graph[node]:
                if (node, nxt) != skip and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    edges = [(a, b) for a in graph for b in graph[a]]
    return sorted(e for e in edges if e[1] not in reachable(e[0], e))


def _ordered(graph: dict[str, set[str]]) -> list[str]:
    """依存される側が先に来る並び。同じ段は置き場の順、名前の順。"""
    order: list[str] = []
    rank = {key: (PLACES.index(_place(key)), key) for key in graph}
    while len(order) < len(graph):
        ready = [k for k in graph if k not in order and graph[k] <= set(order)]
        order.append(min(ready, key=rank.__getitem__))
    return order


def _node(key: str) -> str:
    """mermaid の節点の名前。"""
    return key.replace("/", "_")


def render() -> str:
    """FILES.md の中身。依存の図と、ファイルごとの docstring。"""
    modules = _modules()
    graph = {k: _imported(k, t, set(modules)) for k, t in modules.items()}
    lines = [
        "# neurosurrogate のファイル",
        "",
        "生成物。手で書き換えない。各ファイルの持つものは、"
        "そのファイルの先頭の docstring が正本で、"
        "`uv run python tests/test_design.py` がここに集める。",
        "",
        "## 依存",
        "",
        "矢印は「import する」向き。実際の import から推移的な辺を除いたもの。",
        "",
        "```mermaid",
        "flowchart RL",
    ]
    for place in PLACES:
        keys = [k for k in sorted(modules) if _place(k) == place]
        indent = "    " if place else "  "
        if place:
            lines.append(f'  subgraph {place}["{place}/"]')
        for key in keys:
            lines.append(f'{indent}{_node(key)}["{key.split("/")[-1]}"]')
        if place:
            lines.append("  end")
    for a, b in _reduced(graph):
        lines.append(f"  {_node(a)} --> {_node(b)}")
    lines.append("```")
    for key in _ordered(graph):
        doc = ast.get_docstring(modules[key]) or ""
        summary, _, rest = doc.partition("\n")
        lines += ["", f"## {key}.py", "", summary]
        if rest.strip():
            lines += ["", "```text", rest.strip("\n"), "```"]
    return "\n".join(lines) + "\n"


def test_every_module_has_a_docstring() -> None:
    for key, tree in _modules().items():
        assert ast.get_docstring(tree), f"{key}.py の先頭に docstring が無い"


def test_files_md_is_fresh() -> None:
    assert FILES.read_text() == render(), (
        "`uv run python tests/test_design.py` で作り直す"
    )


if __name__ == "__main__":
    # treefmt は編集したファイルを引数に足して呼ぶが、どれを編集しても全体を作り直す。
    del sys.argv[1:]
    FILES.write_text(render())
