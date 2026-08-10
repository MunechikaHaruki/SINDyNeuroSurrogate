"""**結果の集まり**を扱う層: 系列 (点軸) × run 軸を張ったキー `SimKey`、その組立
(`run_results`)、そこから描画/指標計算に必要な部分集合を取り出す選択操作。
marimo/mlflow 非依存。

**run 軸を持ち込むのはここ**: `runs.EvalSeries` が持つ surrogate は 1 つで run_id を
知らない。カタログ (`eval.SERIES`) に run ごとの surrogate を載せた系列を組んで回し、
`(系列名, 点 index, run_id)` のキーへ詰めるのがこのモジュールで、以降の group/filter
も同じキーの上で行う。

**軸を束ねる型を持たない**: 束ねて見る側 (metrics) が dict を舐めて group/filter
するだけ。
"""

from __future__ import annotations

from collections import Counter

from ..eval import EvalSeries, SimResult
from ..surrogate.bundle import SurrogateBundle

# (系列名, 点 index, run_id)。run_id=None は原系。**軸ごとに 1 要素**で、点軸
# (電流パラメータ) と run 軸を 1 つの文字列へ潰さない。
SimKey = tuple[str, int, str | None]


def run_results(
    catalog: dict[str, dict], bundles: dict[str, SurrogateBundle]
) -> dict[SimKey, SimResult]:
    """カタログ (`eval.SERIES` = 系列名 → `EvalSeries` の構築引数) の全系列 ×
    (原系 + 置換できる run) を回して 1 枚の `SimKey → SimResult` にする
    (**条件 → 結果の唯一の入口**)。

    **run 軸を掛けるのはここ**: run ごとに surrogate を載せた系列を組んで回す。
    1 本も置換できない系列は落とし (回しても比較対象が無い)、原系は `run_id=None`
    として同じ経路 (特別扱いしない)。"""
    out: dict[SimKey, SimResult] = {}
    for name, kwargs in catalog.items():
        original = EvalSeries(**kwargs)
        run_ids = [rid for rid, b in bundles.items() if original.replaceable(b.meta)]
        if not run_ids:
            continue
        for run_id in (None, *run_ids):
            series = (
                original
                if run_id is None
                else EvalSeries(**kwargs, surrogate=bundles[run_id])
            )
            out.update({(name, i, run_id): r for i, r in enumerate(series.simulate())})
    return out


def series(results: dict[SimKey, SimResult]) -> list[str]:
    """系列名 (キーの第 1 軸。掃引しても不変) を最初に現れた順で並べたもの。
    掃引したエントリは複数の点 index が同じ名前へ集まる。"""
    return list(dict.fromkeys(name for name, _point, _run_id in results))


def points_of(results: dict[SimKey, SimResult], name: str) -> list[int]:
    """`name` の系列に属する点 index を掃引点の**値**順に並べたもの (掃引が無ければ
    1 件)。以降の添字はこの並びで、`(name, point, run_id)` でキーを組み立てる。"""
    points = {point for n, point, _run_id in results if n == name}
    return sorted(points, key=lambda p: results[(name, p, None)].point or 0.0)


def run_ids_of(results: dict[SimKey, SimResult], name: str) -> list[str]:
    """`name` の系列で使われた run_id (原系を除く。与えた順)。"""
    return list(
        dict.fromkeys(
            run_id for n, _point, run_id in results if n == name and run_id is not None
        )
    )


def run_names(bundles: dict[str, SurrogateBundle]) -> dict[str, str]:
    """run_id → 表示名 (凡例/行見出し)。**表示名は結果でなく surrogate 側から解く**
    (結果は run_id という同一性だけを持つ)。

    `meta.label` は学習構造 + 学習データまでしか区別しない → library_specs 違いや
    同 config の再実行は同じ label になるため、衝突したものにだけ与えた順の連番を
    付けて潰れを防ぐ (選択を拒否せず全部見せる)。
    """
    labels = [b.meta.label for b in bundles.values()]
    counts = Counter(labels)
    seen: Counter[str] = Counter()
    out: dict[str, str] = {}
    for run_id, label in zip(bundles, labels, strict=True):
        seen[label] += 1
        out[run_id] = label if counts[label] == 1 else f"{label}#{seen[label]}"
    return out


def pair(
    results: dict[SimKey, SimResult], name: str, point: int, run_id: str
) -> tuple[SimResult, SimResult]:
    """(原系, 置換系) の 1 組。"""
    return results[(name, point, None)], results[(name, point, run_id)]


def sources_of(results: dict[SimKey, SimResult], name: str) -> tuple[str, ...]:
    """`name` の系列を描くのに読んだ評価 run の id (回した直後で未保存の結果は
    出所を持たない = 除く。重複除去は順序を保ったまま)。"""
    return tuple(
        dict.fromkeys(
            r.eval_run_id
            for (n, _point, _run_id), r in results.items()
            if n == name and r.eval_run_id is not None
        )
    )
