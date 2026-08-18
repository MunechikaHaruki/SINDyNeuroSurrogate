"""成果物の値と、そのファイル保存。MLflow には依存しない。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure


@dataclass(frozen=True)
class Artifact:
    """成果物 1 件 = 名前 (拡張子抜き) + 図・表・素の値のどれか。

    できるのは**自分を 1 つ書き出すこと**だけ。どの段へ置くかは知らない (置き場を
    決めるのは呼ぶ側)。
    """

    name: str
    obj: Figure | pd.DataFrame | dict[str, Any]

    def save(self, path: Path) -> None:
        """`path` 直下へ 1 件書く。**拡張子は中身が決める** = 表なら CSV、図なら PNG、
        素の値なら JSON (呼ぶ側は綴りを選ばない)。"""
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(self.obj, pd.DataFrame):
            self.obj.to_csv(path / f"{self.name}.csv")
        elif isinstance(self.obj, Figure):
            self.obj.savefig(path / f"{self.name}.png")
        else:
            (path / f"{self.name}.json").write_text(
                json.dumps(self.obj, ensure_ascii=False, indent=2, default=str) + "\n"
            )


@dataclass(frozen=True)
class Artifacts:
    """名前付き成果物の immutable な集合。まとめて 1 つの `path` へ書ける。"""

    items: tuple[Artifact, ...]

    def __iter__(self) -> Iterator[Artifact]:
        return iter(self.items)

    def save(self, path: Path) -> None:
        """全成果物を `path` 直下へ保存する。どの `path` へ置くかは呼ぶ側が決める。"""
        for artifact in self:
            artifact.save(path)
