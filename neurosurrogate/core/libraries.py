from collections.abc import Callable
from dataclasses import dataclass

import pysindy as ps

from neurosurrogate.metrics.opcost import OpCost


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
        from ..registry.feature_libraries import FIXED_LIB_ENTRIES, VARIADIC_LIB_ENTRIES

        def _resolve(spec: dict) -> SubLibrary:
            t = spec["type"]
            inputs = spec["inputs"]
            if t in FIXED_LIB_ENTRIES:
                entries = FIXED_LIB_ENTRIES[t]
                expected = entries[0].func.__code__.co_argcount
                if len(inputs) != expected:
                    raise ValueError(
                        f"type={t!r} は arity={expected} 要求、"
                        f"inputs={inputs} (arity={len(inputs)})"
                    )
                return SubLibrary(entries=entries, inputs=inputs)
            if t in VARIADIC_LIB_ENTRIES:
                return SubLibrary(
                    entries=VARIADIC_LIB_ENTRIES[t](len(inputs)), inputs=inputs
                )
            known = sorted(
                list(FIXED_LIB_ENTRIES.keys()) + list(VARIADIC_LIB_ENTRIES.keys())
            )
            raise ValueError(f"未知 library type: {t!r}。対応 type: {known}")

        subs = [_resolve(s) for s in library_specs]
        return FeatureLibrary(
            sub_libraries=subs,
            library=ps.GeneralizedLibrary(
                [sl.to_ps_library() for sl in subs],
                inputs_per_library=[sl.inputs for sl in subs],
            ),
        )
