from collections.abc import Callable
from dataclasses import dataclass

import pysindy as ps

from neurosurrogate.opcost import OpCost


@dataclass(frozen=True)
class LibraryEntry:
    func: Callable
    name_func: Callable
    cost: OpCost

    def to_cost_entry(self, input_names: list[str]) -> tuple[str, OpCost]:
        return self.name_func(*input_names), self.cost


@dataclass(frozen=True)
class SubLibrary:
    """1 library_spec 単位。entries + 入力インデックス binding。"""

    entries: list[LibraryEntry]
    inputs: list[int]

    def to_ps_library(self) -> ps.CustomLibrary:
        return ps.CustomLibrary(
            library_functions=[e.func for e in self.entries],
            function_names=[e.name_func for e in self.entries],
        )

    def to_cost_dict(self, input_names: list[str]) -> dict[str, OpCost]:
        bound = [input_names[i] for i in self.inputs]
        return dict(e.to_cost_entry(bound) for e in self.entries)


@dataclass(frozen=True)
class FeatureLibrary:
    sub_libraries: list[SubLibrary]
    library: ps.GeneralizedLibrary

    def to_base_cost(self, input_names: list[str]) -> dict[str, OpCost]:
        base_cost: dict[str, OpCost] = {}
        for sl in self.sub_libraries:
            new_data = sl.to_cost_dict(input_names)
            if dup := base_cost.keys() & new_data.keys():
                raise KeyError(f"library間で feature名 重複: {sorted(dup)}")
            base_cost |= new_data
        return base_cost

    @staticmethod
    def build(library_specs: list[dict]) -> "FeatureLibrary":
        from ..registry.feature_libraries import LIB_ENTRIES

        def _resolve(spec: dict) -> SubLibrary:
            inputs = spec["inputs"]
            key = (spec["type"], len(inputs))
            entries = LIB_ENTRIES.get(key)
            if entries is None:
                raise ValueError(
                    f"未対応 library spec: type={key[0]!r}, arity={key[1]}。"
                    f"対応キー: {sorted(LIB_ENTRIES.keys())}"
                )
            return SubLibrary(entries=entries, inputs=inputs)

        subs = [_resolve(s) for s in library_specs]
        return FeatureLibrary(
            sub_libraries=subs,
            library=ps.GeneralizedLibrary(
                [sl.to_ps_library() for sl in subs],
                inputs_per_library=[sl.inputs for sl in subs],
            ),
        )
