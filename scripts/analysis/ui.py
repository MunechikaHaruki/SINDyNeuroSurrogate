from __future__ import annotations

import typing
from pathlib import Path
from typing import Literal

import marimo as mo
import matplotlib.pyplot as plt
import pandas as pd
from analysis.mode import single as analysis_single
from analysis.mode import sweep as analysis_sweep
from mlflow_io import setup_mlflow

from neurosurrogate.currents import CURRENT_MAP
from neurosurrogate.models import MCMODELS

CurrentList: list = list(CURRENT_MAP.keys())
MplStyle = Literal["paper", "presentation"]
MCNameList = list(MCMODELS.keys())

setup_mlflow()


# ---------------------------------------------------------------------------
# Base UI
# ---------------------------------------------------------------------------


def make_base_ui(
    runs_df: pd.DataFrame, target_model: dict[str, list[str]]
) -> mo.ui.dictionary:
    # 学習 train_model × 適用候補 target_model の有効ペアを 1 dropdown で列挙。
    # 未登録 train_model は自身のみを候補とする。label→(train,target) を .value で得る。
    pairs = {
        f"{train}→{tgt}": (train, tgt)
        for train in sorted(runs_df["train_model"].unique())
        for tgt in target_model.get(train, [train])
    }
    plt_options = list(typing.get_args(MplStyle))
    return mo.ui.dictionary(
        {
            "plt_style": mo.ui.radio(options=plt_options, value=plt_options[1]),
            "sim_current_type": mo.ui.dropdown(CurrentList, value="lin&steady&pulse"),
            "dt": mo.ui.number(value=0.01, step=0.001),
            "model_pair": mo.ui.dropdown(
                options=pairs,
                value=next(iter(pairs)),
                label="モデルペア (train→target)",
            ),
        }
    )


def target_of(base_ui: mo.ui.dictionary) -> str:
    """選択ペアの適用先 MC モデル名。"""
    return str(base_ui["model_pair"].value[1])


def train_of(base_ui: mo.ui.dictionary) -> str:
    """選択ペアの学習元 train_model 名。"""
    return str(base_ui["model_pair"].value[0])


# ---------------------------------------------------------------------------
# Setting UI (集約)
# ---------------------------------------------------------------------------


def setup_mpl(matplotlib_style: str):
    style_dir = Path(__file__).resolve().parents[1] / "conf" / "style"
    plt.style.use(style_dir / "base.mplstyle")
    plt.style.use(style_dir / f"{matplotlib_style}.mplstyle")


def _run_selector(
    runs: pd.DataFrame, label: str, selection: Literal["single", "multi"] = "multi"
) -> mo.ui.table:
    return mo.ui.table(
        runs,
        label=label,
        selection=selection,
        initial_selection=[0] if len(runs) else [],
    )


def make_setting_ui(
    runs_df: pd.DataFrame,
    base_ui: mo.ui.dictionary,
    sweep_defaults: analysis_sweep.SweepDefaults,
) -> mo.ui.dictionary:
    current_type = str(base_ui["sim_current_type"].value)
    # 選択 train_model の run のみを提示。run_selector は sim/sweep 各キーへ個別に
    # 埋め、single (1件必須) と sweep (複数可) で選択状態を分離する。
    runs = pd.DataFrame(
        runs_df[runs_df["train_model"] == train_of(base_ui)][
            ["tags.mlflow.runName", "run_id"]
        ]
    )
    d: dict = {
        "sim": analysis_single.make_sim_ui(
            current_type, _run_selector(runs, "single Run (1件)", "single")
        ),
        "run_sim": mo.ui.run_button(label="single 実行"),
    }
    sweep = analysis_sweep.make_sweep_ui(
        current_type, sweep_defaults, _run_selector(runs, "sweep Run (複数可)")
    )
    if sweep is not None:
        d["sweep"] = sweep
        d["run_sweep"] = mo.ui.run_button(label="sweep 実行")
    return mo.ui.dictionary(d)


# ---------------------------------------------------------------------------
# Draw setttings
# ---------------------------------------------------------------------------


def make_draw_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    net = MCMODELS[target_of(base_ui)]
    d: dict = {
        # 既定=soma (全モデルが細胞体を "soma" と命名する共通規約)。
        "eval_comp": mo.ui.dropdown(
            options=net.names, value="soma", label="評価対象comp"
        ),
        "single": analysis_single.make_draw_ui(),
    }
    sweep = analysis_sweep.make_draw_ui(base_ui)
    if sweep is not None:
        d["sweep"] = sweep
    return mo.ui.dictionary(d)
