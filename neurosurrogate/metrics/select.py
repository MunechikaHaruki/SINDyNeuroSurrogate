"""フラットな評価結果 (`dict[SimKey, SimResult]`) から描画/指標計算に必要な部分集合を
取り出す選択操作。**軸を束ねる型を持たない代わりに、束ねて見る側 (metrics) がここで
group/filter する** (eval 側の関心にしない)。marimo/mlflow 非依存。
"""

from __future__ import annotations

from ..runs import SimKey, SimResult


def series(results: dict[SimKey, SimResult]) -> list[str]:
    """系列名 (`SimResult.series`。掃引しても不変) を最初に現れた順で並べたもの。
    掃引したエントリは複数 label (`name#0`, `name#1`, ...) が同じ名前へ集まる。"""
    seen: list[str] = []
    for _label, result in results.items():
        name = result.series
        if name not in seen:
            seen.append(name)
    return seen


def labels_of(results: dict[SimKey, SimResult], name: str) -> list[str]:
    """`name` の系列に属する label を掃引点の値順に並べたもの (掃引が無ければ 1 件)。"""
    labels = {
        label for label, run_id in results if results[(label, run_id)].series == name
    }
    return sorted(labels, key=lambda label: results[(label, None)].point or 0.0)


def run_ids_of(results: dict[SimKey, SimResult], name: str) -> list[str]:
    """`name` の系列で使われた run_id (原系を除く。与えた順)。"""
    seen: list[str] = []
    for label, run_id in results:
        if run_id is None or results[(label, run_id)].series != name:
            continue
        if run_id not in seen:
            seen.append(run_id)
    return seen


def run_label_of(results: dict[SimKey, SimResult], name: str, run_id: str) -> str:
    """表示名 (凡例/行見出し)。原系は "Original" 固定。"""
    label = labels_of(results, name)[0]
    return results[(label, run_id)].run_label or "Original"


def pair(
    results: dict[SimKey, SimResult], label: str, run_id: str
) -> tuple[SimResult, SimResult]:
    """(原系, 置換系) の 1 組。"""
    return results[(label, None)], results[(label, run_id)]


def sources_of(results: dict[SimKey, SimResult], name: str) -> tuple[str, ...]:
    """`name` の系列が参照した出所 (評価 run。読込元が無い = 実行直後の結果は除く。
    重複除去は順序を保ったまま)。"""
    return tuple(
        dict.fromkeys(
            str(r.source)
            for r in results.values()
            if r.series == name and r.source is not None
        )
    )
