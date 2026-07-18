from __future__ import annotations

import typing
from pathlib import Path
from typing import Literal, cast

import marimo as mo
import matplotlib.pyplot as plt
import pandas as pd
from analysis import single as analysis_single
from analysis import sweep as analysis_sweep
from analysis.panel import (
    Panel,
    SaveEntry,
    entry,
    pair,
)
from mlflow_io import (
    load_surrogate_model,
    setup_mlflow,
)

from neurosurrogate.currents import CURRENT_MAP
from neurosurrogate.metrics.eval import EvalResult
from neurosurrogate.models import MCMODELS
from neurosurrogate.surrogate.ansatz import NeuroSurrogateBase
from neurosurrogate.surrogate.replace import replaced_names
from neurosurrogate.view.model import view_model, view_neuron_graph
from neurosurrogate.view.utils import current_preview_fig

# 選択 run 1件 = (run_id, run_name, surrogate)
LoadedRun = tuple[str, str, NeuroSurrogateBase]

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
# Calc
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
# Draw setttings
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


# ---------------------------------------------------------------------------
# Result View
# ---------------------------------------------------------------------------


def view_single(
    run: LoadedRun | None,
    base_ui: mo.ui.dictionary,
    res: EvalResult | None,
    draw_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    """NeuronGraph → 係数 heatmap (選択 run 静的) → single 波形評価 (res ゲート)。"""
    pair_tag = pair(base_ui)
    panel = Panel()
    if run is None:
        panel.note("(single Run 未選択)")
        return panel.done()

    net = MCMODELS[target_of(base_ui)]
    surr = run[2]
    panel.figs(
        "NeuronGraph",
        [
            (
                f"neurograph({pair_tag})",
                view_neuron_graph(net, replaced_names(surr, net)),
            )
        ],
    )
    panel.figs("SINDy 係数", [(f"model({pair_tag})", view_model(surr.sindy_bundle))])

    if res is None:
        panel.note("(single結果なし)")
        return panel.done()

    eval_comp = str(draw_ui["eval_comp"].value)
    suffix = f"{pair_tag},{eval_comp}"
    html, figs, dfs = analysis_single.view_result(draw_ui["single"], res, eval_comp)
    panel.section("波形評価 (single)", html)
    for name, fig in figs:
        panel.save(f"{name}({suffix})", fig)
    panel.save(f"metrics({suffix})", dfs["metrics"])
    panel.save(f"metrics_scalar({suffix})", dfs["metrics(scalar)"])
    return panel.done()


def _eval_df(loaded: list[LoadedRun]) -> pd.DataFrame:
    rows = [
        {
            "run_name": run_name,
            "run_id": rid[:8],
            **surrogate.metrics(),
        }
        for rid, run_name, surrogate in loaded
    ]
    return pd.DataFrame(rows).set_index("run_name")


def view_sweep(
    loaded: list[LoadedRun],
    base_ui: mo.ui.dictionary,
    res: dict | None,
    draw_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    """先頭に評価サマリ表 (選択 run 静的) → 続けて sweep 結果 (res ゲート)。"""
    pair_tag = pair(base_ui)
    summary = _eval_df(loaded)
    panel = Panel()
    panel.section("評価サマリ (preprocessor / OpCost)", summary)
    panel.save(f"eval_summary({pair_tag})", summary)

    if res is None or "sweep" not in draw_ui:
        panel.note("(sweep結果なし)")
        return panel.done()

    eval_comp = str(draw_ui["eval_comp"].value)
    ylim_ui = draw_ui["sweep"]["ylim"]
    ylim = (
        None
        if ylim_ui["auto"].value
        else (float(ylim_ui["min"].value), float(ylim_ui["max"].value))
    )
    trace_html, trace_fig = analysis_sweep.plot_sweep_traces(res, eval_comp)
    panel.section("Sweep 波形 (列=amp / 行=各run vs orig)", trace_html)
    panel.save(f"sweep_traces({pair_tag},{eval_comp})", trace_fig)

    html, fig = analysis_sweep.plot_sweep(
        res,
        eval_comp_name=eval_comp,
        metric_key=draw_ui["sweep"]["metric"].value,
        ylim=ylim,
    )
    panel.section("Sweep メトリクス", html)
    panel.save(f"sweep({pair_tag},{eval_comp})", fig)
    return panel.done()


def plot_preview(
    base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary
) -> tuple[mo.Html, list[SaveEntry]]:
    current_type = str(base_ui["sim_current_type"].value)
    fig = current_preview_fig(
        current_type,
        float(base_ui["dt"].value),
        setting_ui["sim"]["current_params"].value or {},
    )
    html = mo.vstack([mo.md("### 電流プレビュー"), mo.mpl.interactive(fig)])
    return html, [entry(f"current({current_type})", fig)]
