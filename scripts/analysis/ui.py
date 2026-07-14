from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import marimo as mo
import matplotlib.pyplot as plt
import pandas as pd
from analysis import single as analysis_single
from analysis import sweep as analysis_sweep
from matplotlib.figure import Figure
from mlflow_io import (
    get_runs_df,
    load_surrogate_model,
    setup_mlflow,
    sole_run_name,
    sole_target_model,
)

from neurosurrogate.currents import CURRENT_MAP
from neurosurrogate.models import MCMODELS
from neurosurrogate.surrogate import SINDyNeuroSurrogate
from neurosurrogate.view.model import view_model, view_neuron_graph

CurrentList: list = list(CURRENT_MAP.keys())
MplStyle = Literal["paper", "presentation"]
MCNameList = list(MCMODELS.keys())

setup_mlflow()

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "docs" / "slide" / "result"


# ---------------------------------------------------------------------------
# Base UI
# ---------------------------------------------------------------------------


def make_base_ui(target_model: dict[str, list[str]]) -> mo.ui.dictionary:
    runs_df = get_runs_df()
    # train_model → 適用候補 target_model を展開 (候補ごとに行を複製)。
    # 未登録 train_model は自身のみを候補とする。
    runs_df["target_model"] = runs_df["train_model"].map(
        lambda m: target_model.get(m, [m])
    )
    runs_df = runs_df.explode("target_model", ignore_index=True)
    plt_options = list(typing.get_args(MplStyle))
    return mo.ui.dictionary(
        {
            "plt_style": mo.ui.radio(options=plt_options, value=plt_options[1]),
            "sim_current_type": mo.ui.dropdown(CurrentList, value="lin&steady&pulse"),
            "dt": mo.ui.number(value=0.01, step=0.001),
            "run_selector": mo.ui.table(
                pd.DataFrame(
                    runs_df[
                        ["tags.mlflow.runName", "run_id", "train_model", "target_model"]
                    ]
                ),
                label="サロゲート選択 (target=適用先MC / train=学習元)",
                selection="multi",
                initial_selection=[0],
            ),
        }
    )


# ---------------------------------------------------------------------------
# Setting UI (集約)
# ---------------------------------------------------------------------------


def setup_mpl(matplotlib_style: str):
    style_dir = Path(__file__).resolve().parents[1] / "conf" / "style"
    plt.style.use(style_dir / "base.mplstyle")
    plt.style.use(style_dir / f"{matplotlib_style}.mplstyle")


def make_setting_ui(
    base_ui: mo.ui.dictionary,
    sweep_defaults: analysis_sweep.SweepDefaults,
) -> mo.ui.dictionary:
    current_type = str(base_ui["sim_current_type"].value)
    # 置換対象は surrogate の学習元 type で自動導出 → targets 選択 UI は不要
    d: dict = {
        "sim": analysis_single.make_sim_ui(current_type),
        "run_sim": mo.ui.run_button(label="single 実行"),
    }
    sweep = analysis_sweep.make_sweep_ui(current_type, sweep_defaults)
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
) -> dict | None:
    if setting_ui["run_sim"].value:
        return analysis_single.calc_eval(base_ui, setting_ui["sim"])
    return None


def calc_sweep(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
) -> dict | None:
    if "run_sweep" in setting_ui and setting_ui["run_sweep"].value:
        return analysis_sweep.calc_sweep(base_ui, setting_ui["sweep"])
    return None


# ---------------------------------------------------------------------------
# Draw setttings
# ---------------------------------------------------------------------------


def make_draw_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    model_name = sole_target_model(cast(pd.DataFrame, base_ui["run_selector"].value))
    comp_names = MCMODELS[model_name].names
    d: dict = {
        "eval_comp": mo.ui.dropdown(
            options=comp_names, value=comp_names[0], label="評価対象comp"
        ),
        "single": analysis_single.make_draw_ui(),
    }
    sweep = analysis_sweep.make_draw_ui(base_ui)
    if sweep is not None:
        d["sweep"] = sweep
    return mo.ui.dictionary(d)


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------


def _eval_df(
    entries: list[tuple[str, str, SINDyNeuroSurrogate]],
) -> pd.DataFrame:
    rows = [
        {
            "run_name": run_name,
            "run_id": rid[:8],
            **surrogate.metrics(),
        }
        for rid, run_name, surrogate in entries
    ]
    return pd.DataFrame(rows).set_index("run_name")


def render_model_info(
    base_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    selected = cast(pd.DataFrame, base_ui["run_selector"].value)
    loaded: list[tuple[str, str, SINDyNeuroSurrogate]] = [
        (rid, run_name, load_surrogate_model(rid))
        for rid, run_name in zip(
            selected["run_id"].tolist(),
            selected["tags.mlflow.runName"].tolist(),
            strict=True,
        )
    ]
    model_name = sole_target_model(selected)

    save_items: list[SaveEntry] = []
    blocks: list[mo.Html] = []
    for rid, run_name, surrogate in loaded:
        fig = view_model(surrogate.sindy_bundle)
        save_items.append(_entry(f"model({rid[:8]})", fig, f"{run_name}({model_name})"))
        blocks.append(
            mo.vstack(
                [
                    mo.md(f"run_id:{rid[:8]}.. &nbsp;&nbsp;　{run_name}"),
                    mo.md(f"{surrogate.sindy_bundle.equations[:40]}"),
                    mo.mpl.interactive(fig),
                ]
            )
        )
    ng_fig = view_neuron_graph(MCMODELS[model_name])
    save_items.append(_entry("neurograph", ng_fig, model_name))

    html = mo.vstack(
        blocks
        + [
            mo.md("### 評価サマリ (preprocessor / OpCost)"),
            _eval_df(loaded),
            mo.md(f"### NeuronGraph: `{model_name}`"),
            mo.mpl.interactive(ng_fig),
        ]
    )
    return html, save_items


# ---------------------------------------------------------------------------
# Result View
# ---------------------------------------------------------------------------


SaveItem = Figure | pd.DataFrame


@dataclass(frozen=True)
class SaveEntry:
    name: str
    obj: SaveItem
    path: str  # default path (docs/slide/result 相対)


def _entry(name: str, obj: SaveItem, prefix: str) -> SaveEntry:
    ext = ".csv" if isinstance(obj, pd.DataFrame) else ".png"
    return SaveEntry(name, obj, f"_{prefix}_{name}{ext}")


def _fmt_current(base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary) -> str:
    ct = base_ui["sim_current_type"].value
    params = setting_ui["sim"]["current_params"].value or {}
    tail = "&".join(f"{k}:{v}" for k, v in params.items())
    return f"{ct}&{tail}" if tail else ct


def _fmt_sweep(base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary) -> str:
    ct = base_ui["sim_current_type"].value
    s = setting_ui["sweep"]
    return (
        f"{ct}_sw{s['amp_start'].value:g}-{s['amp_stop'].value:g}"
        f"n{s['amp_steps'].value}"
    )


def view_single(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
    res: dict | None,
    draw_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    if res is None:
        return mo.md("(single結果なし)"), []
    selected = cast(pd.DataFrame, base_ui["run_selector"].value)
    model_tag = f"{sole_run_name(selected)}({sole_target_model(selected)})"
    eval_comp = str(draw_ui["eval_comp"].value)
    single_prefix = f"{model_tag}(eval:{eval_comp})_{_fmt_current(base_ui, setting_ui)}"

    html, figs, dfs = analysis_single.view_result(draw_ui["single"], res, eval_comp)
    entries = [_entry(name, fig, single_prefix) for name, fig in figs.items()]
    entries.append(_entry("metrics", dfs["metrics"], single_prefix))
    entries.append(_entry("metrics(scalar)", dfs["metrics(scalar)"], single_prefix))
    return html, entries


def view_sweep(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
    res: dict | None,
    draw_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    if res is None or "sweep" not in draw_ui:
        return mo.md("(sweep結果なし)"), []
    selected = cast(pd.DataFrame, base_ui["run_selector"].value)
    model_tag = f"{sole_run_name(selected)}({sole_target_model(selected)})"
    eval_comp = str(draw_ui["eval_comp"].value)

    ylim_ui = draw_ui["sweep"]["ylim"]
    ylim = (
        None
        if ylim_ui["auto"].value
        else (float(ylim_ui["min"].value), float(ylim_ui["max"].value))
    )
    html, fig = analysis_sweep.plot_sweep(
        res,
        eval_comp_name=eval_comp,
        metric_key=draw_ui["sweep"]["metric"].value,
        ylim=ylim,
    )
    prefix = f"{model_tag}(eval:{eval_comp})_{_fmt_sweep(base_ui, setting_ui)}"
    return html, [_entry("sweep", fig, prefix)]


def plot_preview(
    base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary
) -> tuple[mo.Html, list[SaveEntry]]:
    fig = analysis_single.plot_current_preview(base_ui, setting_ui["sim"])
    html = analysis_single.render_current_preview(fig)
    return html, [_entry("current", fig, _fmt_current(base_ui, setting_ui))]


# ---------------------------------------------------------------------------
# Save Panel
# ---------------------------------------------------------------------------


SAVERS: dict[type, typing.Callable[[typing.Any, Path], None]] = {
    Figure: lambda o, p: o.savefig(p, dpi=300, bbox_inches="tight"),
    pd.DataFrame: lambda o, p: o.to_csv(p),
}


def make_save_panel(entries: list[SaveEntry]) -> mo.ui.dictionary:
    """SaveEntry から各 name の path入力＋保存ボタンを生成。"""
    return mo.ui.dictionary(
        {
            e.name: mo.ui.dictionary(
                {
                    "path": mo.ui.text(value=e.path, label=e.name),
                    "save": mo.ui.run_button(label=f"{e.name} 保存"),
                }
            )
            for e in entries
        }
    )


def render_save_panel(panel: mo.ui.dictionary) -> mo.Html:
    rows = [
        mo.hstack(
            [item["path"], item["save"]],
            justify="start",
        )
        for item in panel.values()
    ]
    return mo.vstack([mo.md("### 画像保存パネル (docs/result/ 配下)"), *rows])


def save(save_panel: mo.ui.dictionary, entries: list[SaveEntry]) -> mo.Html:
    msgs: list[mo.Html] = []
    for e in entries:
        ctrl = save_panel[e.name]
        if not ctrl["save"].value:
            continue
        path = RESULT_DIR / str(ctrl["path"].value).strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        SAVERS[type(e.obj)](e.obj, path)
        msgs.append(mo.md(f"✅ {e.name}: `{path.relative_to(REPO_ROOT)}`"))
    return mo.vstack(msgs) if msgs else mo.md("(未保存)")
