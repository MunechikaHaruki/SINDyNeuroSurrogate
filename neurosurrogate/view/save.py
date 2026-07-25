"""成果物 (図 / 表) の運搬と永続化。marimo 非依存。

view が返す `(id, fig)` 列に**保存名**を与えた `SaveEntry` が、表示と保存の共通の
運搬形になる (同じ列を UI は描画に、保存は書き出しに流す = 表示と保存が食い違わない)。
何を書けるか (`SaveItem`)・拡張子・書き出し方 (`SAVERS`) と meta.json の同梱まで
ここが持ち、UI 側は「どれを選んだか」「どこへ」だけを渡す。
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

SaveItem = Figure | pd.DataFrame


@dataclass(frozen=True)
class SaveEntry:
    name: str
    obj: SaveItem
    path: str  # 保存先ディレクトリからの相対パス


def entry(name: str, obj: SaveItem) -> SaveEntry:
    """name をそのまま既定ファイル名に (拡張子のみ付与)。呼び出し側が pair 等を含む
    最終的な表示名を組む。"""
    ext = ".csv" if isinstance(obj, pd.DataFrame) else ".png"
    return SaveEntry(name, obj, f"{name}{ext}")


def flatten(groups: dict[str, list[SaveEntry]]) -> list[SaveEntry]:
    """グループ (model/single/sweep) を平坦化 (表示はグループ分割・保存は一括)。"""
    return [e for es in groups.values() for e in es]


SAVERS: dict[type, Callable[[Any, Path], None]] = {
    Figure: lambda o, p: o.savefig(p, dpi=300, bbox_inches="tight"),
    pd.DataFrame: lambda o, p: o.to_csv(p),
}


def save_entries(
    entries: list[SaveEntry],
    dest: Path,
    meta: dict,
    names: set[str] | None = None,
) -> list[Path]:
    """entry を `dest` 直下へ書き出し、`meta.json` (再現用の設定 snapshot) を同階層に
    置く。names=None で全部、指定時はその name だけ。返り値は書いたパス列 = 呼び出し
    側は表示に流すだけ (何を書いたかの判断はここが済ませる)。"""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str)
    )
    saved = []
    for e in entries:
        if names is not None and e.name not in names:
            continue
        path = dest / e.path
        # entry 名は `<spec ラベル>/<図名>` のように階層を持つ (適用先ごとに束ねる) →
        # 保存側もその階層をディレクトリとして掘る。
        path.parent.mkdir(parents=True, exist_ok=True)
        SAVERS[type(e.obj)](e.obj, path)
        saved.append(path)
    return saved
