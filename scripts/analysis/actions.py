from __future__ import annotations

from typing import cast

import marimo as mo
import pandas as pd
from analysis import single as analysis_single
from analysis import sweep as analysis_sweep
from mlflow_io import load_surrogate_model

from neurosurrogate.metrics.eval import EvalResult
from neurosurrogate.surrogate.ansatz import NeuroSurrogateBase

# 選択 run 1件 = (run_id, run_name, surrogate)
LoadedRun = tuple[str, str, NeuroSurrogateBase]


# ---------------------------------------------------------------------------
# Calc (run ボタンゲート → 各 ansatz へ委譲)
# ---------------------------------------------------------------------------


def calc_single(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
) -> EvalResult | None:
    if setting_ui["run_sim"].value:
        return analysis_single.calc_eval(base_ui, setting_ui)
    return None


def calc_sweep(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
) -> dict | None:
    if "run_sweep" in setting_ui and setting_ui["run_sweep"].value:
        return analysis_sweep.calc_sweep(base_ui, setting_ui)
    return None


# ---------------------------------------------------------------------------
# Load (run_selector 選択 → surrogate ロード)
# ---------------------------------------------------------------------------


def load_selected(sub_ui: mo.ui.dictionary) -> list[LoadedRun]:
    """sub_ui の run_selector 選択 run の surrogate をロードし (id, name, surrogate)
    列で返す。sweep (複数可) 用。single は `load_single`。"""
    selected = cast(pd.DataFrame, sub_ui["run_selector"].value)
    return [
        (rid, run_name, load_surrogate_model(rid))
        for rid, run_name in zip(
            selected["run_id"].tolist(),
            selected["tags.mlflow.runName"].tolist(),
            strict=True,
        )
    ]


def load_single(sub_ui: mo.ui.dictionary) -> LoadedRun | None:
    """single 用。run_selector (単一選択) の 0/1 件を 1 run or None に畳む。"""
    loaded = load_selected(sub_ui)
    return loaded[0] if loaded else None
