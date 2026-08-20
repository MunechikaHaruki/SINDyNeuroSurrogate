"""名前付き surrogate 列と、評価系列への適用範囲。"""

from collections.abc import Iterator
from dataclasses import dataclass

from ..sim.spec import EvalSeries
from .model import Surrogate


@dataclass(frozen=True)
class SurrogateRuns:
    """一意な run 名と選択順を保った surrogate 列。"""

    runs: tuple[tuple[str, Surrogate], ...]

    def __post_init__(self) -> None:
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError(f"学習 run 名が重複 {names}")
        for name in names:
            if not name or name in (".", "..") or any(c in name for c in "/\\\0"):
                raise ValueError(f"学習 run 名をpathに使えない {name!r}")

    def __iter__(self) -> Iterator[tuple[str, Surrogate]]:
        return iter(self.runs)

    def __len__(self) -> int:
        return len(self.runs)

    @property
    def names(self) -> tuple[str, ...]:
        """選択順の学習 run 名。"""
        return tuple(name for name, _ in self.runs)

    def replacing(self, series: EvalSeries) -> "SurrogateRuns":
        """この掃引が指定した置換対象を**全部**置換できる run だけに絞る。"""
        return SurrogateRuns(
            tuple(
                (name, surrogate)
                for name, surrogate in self
                if surrogate.spec.applicable(series)
            )
        )

    def surrogate(self, name: str) -> Surrogate:
        """学習 run 名から surrogate を引く。"""
        for candidate, surrogate in self:
            if candidate == name:
                return surrogate
        raise KeyError(f"run {name!r} がsurrogate列に無い")
