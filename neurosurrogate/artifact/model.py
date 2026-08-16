"""保存方式に依存しない成果物と、その集合。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
from matplotlib.figure import Figure


@dataclass(frozen=True)
class Artifact:
    """成果物 1 件 = 保存段で使う名前 + 図または表。"""

    name: str
    obj: Figure | pd.DataFrame


@dataclass(frozen=True)
class Artifacts:
    """名前付き成果物の immutable な集合。保存先や保存方式は知らない。"""

    items: tuple[Artifact, ...]

    def __iter__(self) -> Iterator[Artifact]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def __add__(self, other: Artifacts) -> Artifacts:
        return Artifacts(self.items + other.items)

    def under(self, path: str) -> Artifacts:
        """全成果物を同じ保存階層の下へ移す。I/O は行わない。"""
        return Artifacts(
            tuple(
                Artifact(f"{path}/{artifact.name}", artifact.obj) for artifact in self
            )
        )
