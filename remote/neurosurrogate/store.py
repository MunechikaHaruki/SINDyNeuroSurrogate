"""run の置き場の学習器とモデル: 学習器を書いて読み、モデルの学習器を組む。

置き場は `runs/parts/<ハッシュ>/` (学習器 1 つ) と `runs/models/<ハッシュ>/` (学習器の
並びを組んだモデル)。どれも spec.json を持ち、学習器の spec は学習の入力を鍵で、モデルの
spec は学習器をハッシュで指す。学習データは学習の入力の元のモデル
`runs/inputs/<鍵>/original/` (単発の入力なので点は 1 つ)。KAM のモデルは学習器を持たず、
群を spec に直に持ち、学習の入力 (図を描く軌道) を鍵で指す。
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


def train_root(runs: Path, key: str) -> Path:
    """学習の入力 `key` の元のモデルの run (学習データ)。"""
    return runs / "inputs" / key / "original"


def _train_data(runs: Path, key: str) -> dict:
    """学習の入力 `key` の電流 (SimSpec の dict)。単発の入力なので点は 1 つ。"""
    spec: dict = json.loads((train_root(runs, key) / "spec.json").read_text())
    data: dict = spec["input"]["spec"]
    return data


def part_config(runs: Path, part: dict) -> Config:
    """学習器の spec を群 1 つの設定にする (学習データの電流は指す学習の入力から)。"""
    data = _train_data(runs, part["input"])
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

    学習器は同じ区画の型・形で、同じ学習の入力・学習区画で学習したものに限る。
    KAM のモデルは群ごとの KAM の縮約を区画の型の式から組む (学習しない)。
    """
    if "kam" in model:
        kam = model["kam"]
        config = Config.from_dict(
            {
                "cell": model["cell"],
                "form": model["form"],
                "data": _train_data(runs, kam["input"]),
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
    if len({(p["input"], p["train_comp"]) for p in parts}) > 1:
        raise ValueError("学習器の学習の入力・学習区画がそろわない")
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
