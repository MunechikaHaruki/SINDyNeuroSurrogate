"""成果物・レポートの値と、そのファイル保存。MLflow には依存しない。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class Tuning:
    """1 レポートの描画条件。

    何を描くかは宣言せず、成果物の種類はモデルと評価結果の形から決まる。
    `eval_comp` だけは適用先の系列を選んだ後で決まるため、既定値を持たない。
    """

    eval_comp: str
    view_comps: tuple[str, ...] = ()
    metric: str = "spike_count"
    detail_point: int = 0
    spike_orig: int = 0
    spike_surr: int = 0
    metric_ylim: tuple[float, float] | None = None


@dataclass(frozen=True)
class Report:
    """描画条件と段ごとの成果物を持つ、保存可能な 1 レポート。"""

    tuning: Tuning
    sections: tuple[tuple[Path, Artifacts], ...]

    def save(self, path: Path) -> tuple[Path, ...]:
        """全成果物と描画条件を `path` 以下へ保存し、書いたファイルを返す。"""
        tuning_path = path / "draw.json"
        tuning_path.parent.mkdir(parents=True, exist_ok=True)
        tuning_path.write_text(
            json.dumps(asdict(self.tuning), ensure_ascii=False, indent=2) + "\n"
        )
        return (
            *(
                file.relative_to(path)
                for directory, artifacts in self.sections
                for file in artifacts.save(path / directory)
            ),
            tuning_path.relative_to(path),
        )
