from __future__ import annotations

from typing import cast

import marimo as mo
import pandas as pd
from analysis.mode import single as analysis_single
from analysis.mode import sweep as analysis_sweep
from mlflow_io import LoadedRun, load_runs

from neurosurrogate.metrics.eval import EvalResult

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
    loaded: list[LoadedRun],
) -> dict | None:
    if "run_sweep" in setting_ui and setting_ui["run_sweep"].value:
        return analysis_sweep.calc_sweep(base_ui, setting_ui, loaded)
    return None


# ---------------------------------------------------------------------------
# Load (run_selector 選択 → surrogate ロード)
# ---------------------------------------------------------------------------


def load_selected(sub_ui: mo.ui.dictionary) -> list[LoadedRun]:
    """run_selector 選択 (複数可) の run_id を抽出し load_runs へ委譲。
    single (1件) は `load_single`。"""
    run_ids = cast(pd.DataFrame, sub_ui["run_selector"].value)["run_id"].tolist()
    return load_runs(run_ids)


def load_single(sub_ui: mo.ui.dictionary) -> LoadedRun | None:
    """single 用。run_selector (単一選択) の 0/1 件を 1 run or None に畳む。"""
    loaded = load_selected(sub_ui)
    return loaded[0] if loaded else None
