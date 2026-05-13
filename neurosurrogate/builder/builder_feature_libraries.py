from dataclasses import dataclass
from typing import Callable

import pysindy as ps

from neurosurrogate.profiler.profiler_model import OpCost


@dataclass(frozen=True)
class LibraryEntry:
    func: Callable
    name_func: Callable
    cost: OpCost


def build_featurelib_and_basecost(library_specs):
    from .registry_feature_libraries import LIB_BUILDER_REGISTRY

    def _entries_to_basecost(
        entries: list[LibraryEntry], inputs_list: list
    ) -> dict[str:OpCost]:
        base_cost_map = {}
        for entry in entries:
            input_names: list = [f"inputs{input_id}" for input_id in inputs_list]
            base_cost_map[f"{entry.name_func(*input_names)}"] = entry.cost
        return base_cost_map

    def _entries_to_library(entries: list[LibraryEntry]) -> ps.CustomLibrary:
        return ps.CustomLibrary(
            library_functions=[e.func for e in entries],
            function_names=[e.name_func for e in entries],
        )

    def _build_one(spec):
        builder = LIB_BUILDER_REGISTRY.get(spec["type"])
        if builder is None:
            raise ValueError(f"未知のlibrary type: {spec['type']}")
        return builder(spec)

    libraries = []
    inputs_per_library = []
    base_cost = {}
    for s in library_specs:
        inputs_list: list = s["inputs"]
        new_entries: list[LibraryEntry] = _build_one(s)
        libraries.append(_entries_to_library(new_entries))

        new_data = _entries_to_basecost(new_entries, inputs_list)
        duplicates = base_cost.keys() & new_data.keys()
        if duplicates:
            detail = "\n".join(
                [
                    f"  - Key: {k}\n    Existing Value: {base_cost[k]}\n    New Value: {new_data[k]}"
                    for k in duplicates
                ]
            )
            raise KeyError(
                f"辞書の結合中にキーの重複が発生しました。上書きを防止します:\n{detail}"
            )
        base_cost |= new_data
        inputs_per_library.append(inputs_list)
    return ps.GeneralizedLibrary(
        libraries, inputs_per_library=inputs_per_library
    ), base_cost
