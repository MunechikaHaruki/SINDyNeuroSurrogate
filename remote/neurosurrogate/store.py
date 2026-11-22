"""run の置き場の学習器とモデル: 学習器を書いて読み、モデルの学習器を組む。

置き場は `runs/data/<ハッシュ>/` (学習データ)、`runs/parts/<ハッシュ>/` (学習器 1 つ)、
`runs/models/<ハッシュ>/` (学習器の並びを組んだモデル)。どれも spec.json を持ち、
学習器の spec は学習データを、モデルの spec は学習器をハッシュで指す。KAM のモデルは
学習器を持たず、群を spec に直に持ち、学習データ (図を描く軌道) をハッシュで指す。
"""

import json
from pathlib import Path
from typing import Any

import joblib

from .surrogate.model import Config, Surrogate, fit_group

PART_FILE = "learner.joblib"


def _spec(runs: Path, top: str, key: str) -> dict:
    spec: dict = json.loads((runs / top / key / "spec.json").read_text())
    return spec


def part_config(runs: Path, part: dict) -> Config:
    """学習器の spec を群 1 つの設定にする (学習データは指す先から)。"""
    data = _spec(runs, "data", part["data"])["data"]
    return Config.from_dict(
        {
            "cell": part["cell"],
            "form": part["form"],
            "data": data,
            "groups": [part["group"]],
            "train_comp": part.get("train_comp"),
        }
    )


def save_part(folder: Path, part: tuple[Any, Any]) -> None:
    joblib.dump(part, folder / PART_FILE)


def assemble(runs: Path, model: dict) -> Surrogate:
    """モデルの spec が並べた学習器を読み、サロゲートに組む。

    学習器は同じ区画の型・形で、同じ学習データ・学習区画で学習したものに限る。
    KAM のモデルは群ごとの KAM の縮約を区画の型の式から組む (学習しない)。
    """
    if "kam" in model:
        kam = model["kam"]
        config = Config.from_dict(
            {
                "cell": model["cell"],
                "form": model["form"],
                "data": _spec(runs, "data", kam["data"])["data"],
                "groups": kam["groups"],
                "train_comp": kam["train_comp"],
                "absorbed": model.get("absorbed", []),
            }
        )
        return Surrogate(config, [fit_group(config, g, None) for g in config.groups])
    parts = [_spec(runs, "parts", key) for key in model["parts"]]
    kinds = {(p["cell"], p["form"]) for p in parts}
    if kinds != {(model["cell"], model["form"])}:
        raise ValueError(f"学習器の区画の型・形がそろわない: {kinds}")
    if len({(p["data"], p["train_comp"]) for p in parts}) > 1:
        raise ValueError("学習器の学習データ・学習区画がそろわない")
    base = part_config(runs, parts[0])
    config = Config.from_dict(
        {
            **base.to_dict(),
            "groups": [p["group"] for p in parts],
            "absorbed": model.get("absorbed", []),
        }
    )
    learned = [joblib.load(runs / "parts" / key / PART_FILE) for key in model["parts"]]
    return Surrogate(config, learned)


def load_model(runs: Path, key: str) -> Surrogate:
    """`runs/models/<key>/` のモデルを組む。"""
    return assemble(runs, _spec(runs, "models", key))
