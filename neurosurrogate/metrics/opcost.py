from dataclasses import dataclass, fields


@dataclass(frozen=True)
class OpCost:
    exp: int = 0
    div: int = 0
    pm: int = 0
    mul: int = 0

    def __add__(self, other: "OpCost") -> "OpCost":
        return OpCost(
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in fields(self)
            }
        )

    def __mul__(self, n: int) -> "OpCost":
        return OpCost(**{f.name: getattr(self, f.name) * n for f in fields(self)})

    def to_dict(self) -> dict[str, int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def diff_dict(self, other: "OpCost | None") -> dict[str, int]:
        if other is None:
            return {}
        surr_d = self.to_dict()
        orig_d = other.to_dict()
        return {
            **{f"cost/surrogate/{k}": v for k, v in surr_d.items()},
            **{f"cost/original/{k}": v for k, v in orig_d.items()},
            **{f"cost/diff/{k}": surr_d[k] - orig_d[k] for k in orig_d},
        }
