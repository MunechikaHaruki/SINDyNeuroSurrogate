from collections.abc import Callable
from dataclasses import dataclass

import pysindy as ps

from neurosurrogate.profiler.profiler_model import OpCost


@dataclass(frozen=True)
class LibraryEntry:
    func: Callable
    name_func: Callable
    cost: OpCost

    def to_cost_entry(self, input_names: list[str]) -> tuple[str, OpCost]:
        return self.name_func(*input_names), self.cost


@dataclass(frozen=True)
class FeatureLibrary:
    library: ps.GeneralizedLibrary
    _entries_with_inputs: list[tuple[list[LibraryEntry], list[int]]]

    def to_base_cost(self, input_names: list[str]) -> dict[str, OpCost]:
        base_cost = {}
        for entries, inputs_list in self._entries_with_inputs:
            new_data = dict(
                entry.to_cost_entry([input_names[i] for i in inputs_list])
                for entry in entries
            )
            duplicates = base_cost.keys() & new_data.keys()
            if duplicates:
                raise KeyError(
                    "辞書の結合中にキーの重複が発生しました。上書きを防止します:\n"
                    + "\n".join(
                        f"  - Key: {k}\n    Existing Value: {base_cost[k]}\n    New Value: {new_data[k]}"  # noqa: E501
                        for k in duplicates
                    )
                )

            base_cost |= new_data
        return base_cost

    @staticmethod
    def build(library_specs: list[dict]) -> "FeatureLibrary":
        from .registry_feature_libraries import LIB_BUILDER_REGISTRY

        def _build_one(spec):
            builder = LIB_BUILDER_REGISTRY.get(spec["type"])
            if builder is None:
                raise ValueError(f"未知のlibrary type: {spec['type']}")
            return builder(spec)

        def _entries_to_library(entries: list[LibraryEntry]) -> ps.CustomLibrary:
            return ps.CustomLibrary(
                library_functions=[e.func for e in entries],
                function_names=[e.name_func for e in entries],
            )

        libraries = []
        inputs_per_library = []
        entries_with_inputs = []

        for s in library_specs:
            new_entries: list[LibraryEntry] = _build_one(s)
            libraries.append(_entries_to_library(new_entries))
            inputs_per_library.append(s["inputs"])
            entries_with_inputs.append((new_entries, s["inputs"]))

        return FeatureLibrary(
            library=ps.GeneralizedLibrary(
                libraries, inputs_per_library=inputs_per_library
            ),
            _entries_with_inputs=entries_with_inputs,
        )
