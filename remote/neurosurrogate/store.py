"""run の置き場の学習器とモデル: 学習器を書いて読み、モデルの学習器を組む。

置き場は `runs/data/<ハッシュ>/` (学習データ)、`runs/parts/<ハッシュ>/` (学習器 1 つ)、
`runs/models/<ハッシュ>/` (学習器の並びを組んだモデル)。どれも spec.json を持ち、
学習器の spec は学習データを、モデルの spec は学習器をハッシュで指す。
"""

import json
from pathlib import Path
from typing import Any

import joblib

from .surrogate.model import Config, Surrogate

PART_FILE = "part.joblib"


def _spec(runs: Path, top: str, key: str) -> dict:
    spec: dict = json.loads((runs / top / key / "spec.json").read_text())
    return spec


def part_config(runs: Path, part: dict) -> Config:
    """学習器の spec を、群 1 つの設定にする (学習データは指す先の spec から)。"""
    data = _spec(runs, "data", part["data"])["data"]
    return Config.from_dict(
        {
            "cell": part["cell"],
            "form": part["form"],
            "data": data,
            "groups": [part["group"]],
            "train_comp": part["train_comp"],
        }
    )


def save_part(folder: Path, part: tuple[Any, Any]) -> None:
    joblib.dump(part, folder / PART_FILE)


def assemble(runs: Path, model: dict) -> Surrogate:
    """モデルの spec が並べた学習器を読み、サロゲートに組む。

    学習器は同じ区画の型・形・学習データ・学習区画で学習したものに限る。
    """
    parts = [_spec(runs, "parts", key) for key in model["parts"]]
    shared = {(p["cell"], p["form"], p["data"], p["train_comp"]) for p in parts}
    if len(shared) != 1 or {(model["cell"], model["form"])} != {s[:2] for s in shared}:
        raise ValueError(
            f"学習器の区画の型・形・学習データ・学習区画がそろわない: {shared}"
        )
    first = part_config(runs, parts[0])
    config = Config.from_dict(
        {
            **first.to_dict(),
            "groups": [p["group"] for p in parts],
            "absorbed": model.get("absorbed", []),
        }
    )
    learned = [joblib.load(runs / "parts" / key / PART_FILE) for key in model["parts"]]
    return Surrogate(config, learned)


def load_model(runs: Path, key: str) -> Surrogate:
    """`runs/models/<key>/` のモデルを組む。"""
    return assemble(runs, _spec(runs, "models", key))
