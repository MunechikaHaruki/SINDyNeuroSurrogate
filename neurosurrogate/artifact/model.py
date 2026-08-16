"""成果物と、そのファイル保存を担う集合。MLflow には依存しない。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


@dataclass(frozen=True)
class Artifact:
    """成果物 1 件 = 保存段で使う名前 + 図または表。"""

    name: str
    obj: Figure | pd.DataFrame


@dataclass(frozen=True)
class Artifacts:
    """名前付き成果物の immutable な集合。PNG/CSV として一括保存できる。"""

    items: tuple[Artifact, ...]

    def __iter__(self) -> Iterator[Artifact]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def save(self, path: Path) -> tuple[Path, ...]:
        """全成果物を `path` 以下へ保存し、書いたファイルを返す。"""
        written: list[Path] = []
        for artifact in self:
            suffix = ".csv" if isinstance(artifact.obj, pd.DataFrame) else ".png"
            target = path / f"{artifact.name}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(artifact.obj, pd.DataFrame):
                artifact.obj.to_csv(target)
            else:
                artifact.obj.savefig(target)
            written.append(target)
        return tuple(written)
