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

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "scripts" / "conf" / "surrogate" / "result"


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


def make_setting_ui(
    runs_df: pd.DataFrame,
    base_ui: mo.ui.dictionary,
    sweep_defaults: analysis_sweep.SweepDefaults,
) -> mo.ui.dictionary:
    current_type = str(base_ui["sim_current_type"].value)
    # 選択 train_model の run のみを提示 (single=1件必須 / sweep=複数可)。
    runs = runs_df[runs_df["train_model"] == train_of(base_ui)]
    # 置換対象は surrogate の学習元 type で自動導出 → targets 選択 UI は不要
    d: dict = {
        "run_selector": mo.ui.table(
            pd.DataFrame(runs[["tags.mlflow.runName", "run_id"]]),
            label="Run 選択 (single=1件 / sweep=複数可)",
            selection="multi",
            initial_selection=[0] if len(runs) else [],
        ),
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


def load_selected(setting_ui: mo.ui.dictionary) -> list[LoadedRun]:
    """選択 run の surrogate を 1 回だけロードし (id, name, surrogate) 列で共有。
    neurograph / heatmap / 評価サマリ の 3 描画がこれを消費 (再 DL 回避)。"""
    selected = cast(pd.DataFrame, setting_ui["run_selector"].value)
    return [
        (rid, run_name, load_surrogate_model(rid))
        for rid, run_name in zip(
            selected["run_id"].tolist(),
            selected["tags.mlflow.runName"].tolist(),
            strict=True,
        )
    ]


def make_draw_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    comp_names = MCMODELS[target_of(base_ui)].names
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


def render_model_info(
    loaded: list[LoadedRun],
    base_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    """model_info は neurograph のみ (heatmap→single / 評価サマリ→sweep へ移管)。"""
    target = target_of(base_ui)
    net = MCMODELS[target]
    figs = [
        (f"neurograph({run_name})", view_neuron_graph(net, replaced_names(surr, net)))
        for _, run_name, surr in loaded
    ]
    save_items = [_entry(name, fig, target) for name, fig in figs]
    bodies = [
        mo.vstack([mo.md(f"##### {name}"), mo.mpl.interactive(fig)])
        for name, fig in figs
    ]
    return mo.vstack([mo.md("### NeuronGraph"), *bodies]), save_items


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
    return SaveEntry(name, obj, f"{prefix}_{name}{ext}")


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
    loaded: list[LoadedRun],
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
    res: EvalResult | None,
    draw_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    """先頭に係数 heatmap (選択 run 静的) → 続けて single 波形評価 (res ゲート)。"""
    target = target_of(base_ui)
    # heatmap は電流非依存のモデル静的図 → prefix=target (current を含めない)。
    heat = [
        (f"model({run_name})", view_model(surr.sindy_bundle))
        for _, run_name, surr in loaded
    ]
    blocks: list[mo.Html] = [mo.md("### SINDy 係数")]
    blocks += [
        part
        for name, fig in heat
        for part in (mo.md(f"##### {name}"), mo.mpl.interactive(fig))
    ]
    entries = [_entry(name, fig, target) for name, fig in heat]

    if res is None:
        blocks.append(mo.md("(single結果なし)"))
        return mo.vstack(blocks), entries

    eval_comp = str(draw_ui["eval_comp"].value)
    run_tag = loaded[0][1] if len(loaded) == 1 else target
    single_prefix = (
        f"{run_tag}({target})(eval:{eval_comp})_{_fmt_current(base_ui, setting_ui)}"
    )
    html, figs, dfs = analysis_single.view_result(draw_ui["single"], res, eval_comp)
    blocks += [mo.md("### 波形評価 (single)"), html]
    entries += [_entry(name, fig, single_prefix) for name, fig in figs]
    entries.append(_entry("metrics", dfs["metrics"], single_prefix))
    entries.append(_entry("metrics(scalar)", dfs["metrics(scalar)"], single_prefix))
    return mo.vstack(blocks), entries


def view_sweep(
    loaded: list[LoadedRun],
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
    res: dict | None,
    draw_ui: mo.ui.dictionary,
) -> tuple[mo.Html, list[SaveEntry]]:
    """先頭に評価サマリ表 (選択 run 静的) → 続けて sweep 結果 (res ゲート)。"""
    target = target_of(base_ui)
    summary = _eval_df(loaded)
    blocks: list[mo.Html] = [mo.md("### 評価サマリ (preprocessor / OpCost)"), summary]
    entries = [_entry("eval_summary", summary, target)]

    if res is None or "sweep" not in draw_ui:
        blocks.append(mo.md("(sweep結果なし)"))
        return mo.vstack(blocks), entries

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
    prefix = f"{target}(eval:{eval_comp})_{_fmt_sweep(base_ui, setting_ui)}"
    blocks += [mo.md("### Sweep"), html]
    entries.append(_entry("sweep", fig, prefix))
    return mo.vstack(blocks), entries


def plot_preview(
    base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary
) -> tuple[mo.Html, list[SaveEntry]]:
    fig = current_preview_fig(
        str(base_ui["sim_current_type"].value),
        float(base_ui["dt"].value),
        setting_ui["sim"]["current_params"].value or {},
    )
    html = mo.vstack([mo.md("### 電流プレビュー"), mo.mpl.interactive(fig)])
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
