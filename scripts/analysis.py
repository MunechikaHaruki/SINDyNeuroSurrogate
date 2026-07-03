import typing
from pathlib import Path
from typing import Literal, cast

import analysis_single
import analysis_sweep
import marimo as mo
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from io_handler import TARGET_EXP, RunInfo, setup_mlflow
from matplotlib.figure import Figure

from neurosurrogate.builder.registry_current import FUNC_MAP
from neurosurrogate.model.registry_neuron import MCMODELS
from neurosurrogate.profiler.profiler_view import view_neuron_graph
from neurosurrogate.profiler.registry_view import DRAW_MAP

CurrentList: list = list(FUNC_MAP.keys())
DRAW_LIST: list = list(DRAW_MAP.keys())
MplStyle = Literal["paper", "presentation"]
MCNameList = list(MCMODELS.keys())

setup_mlflow()

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "docs" / "slide" / "result"


# ---------------------------------------------------------------------------
# Base UI
# ---------------------------------------------------------------------------


def _get_runs_df():
    experiment = mlflow.get_experiment_by_name(TARGET_EXP)
    if experiment is None:
        raise ValueError(
            f"Experiment '{TARGET_EXP}' が見つかりません。名前を確認してください。"
        )
    all_runs_df = cast(
        pd.DataFrame, mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    )
    if all_runs_df.empty:
        raise ValueError(f"Experiment '{TARGET_EXP}' にrunが存在しません。")
    runs_df = all_runs_df.copy()
    runs_df = runs_df.sort_values("start_time", ascending=False)
    runs_df["start_time"] = runs_df["start_time"].dt.strftime("%m-%d %H:%M:%S")
    runs_df = runs_df[
        ["tags.mlflow.runName", "run_id", "start_time"]
        + [c for c in runs_df.columns if "metrics" in c or "params" in c]
    ]
    return runs_df


def make_base_ui() -> mo.ui.dictionary:
    runs_df = _get_runs_df()
    plt_options = list(typing.get_args(MplStyle))
    return mo.ui.dictionary(
        {
            "plt_style": mo.ui.radio(options=plt_options, value=plt_options[1]),
            "sim_current_type": mo.ui.dropdown(CurrentList, value="lin_single_pulse"),
            "model_name": mo.ui.dropdown(
                options=list(MCMODELS.keys()),
                value="hh",
            ),
            "dt": mo.ui.number(value=0.01, step=0.001),
            "run_selector": mo.ui.table(
                pd.DataFrame(runs_df[["tags.mlflow.runName", "run_id"]]),
                label="Run 選択 (single: 1件のみ / sweep: 複数可)",
                selection="multi",
                initial_selection=[0],
            ),
        }
    )


def setup_mpl(matplotlib_style: str):
    style_dir = Path(__file__).resolve().parent / "conf" / "style"
    plt.style.use(style_dir / "base.mplstyle")
    plt.style.use(style_dir / f"{matplotlib_style}.mplstyle")


# ---------------------------------------------------------------------------
# Draw / Surrogate Targets UI
# ---------------------------------------------------------------------------


def _make_surrogate_targets_ui(base_ui: mo.ui.dictionary) -> mo.ui.multiselect:
    comp_names = MCMODELS[str(base_ui["model_name"].value)].names
    return mo.ui.multiselect(
        options=comp_names, value=[comp_names[0]], label="surrogate targets"
    )


def make_draw_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    comp_names = MCMODELS[str(base_ui["model_name"].value)].names
    return mo.ui.dictionary(
        {
            "eval_comp": mo.ui.dropdown(options=comp_names, value=comp_names[0]),
            "draw_func": mo.ui.dropdown(options=DRAW_LIST, value=DRAW_LIST[0]),
        }
    )


def render_draw_ui(draw_ui: mo.ui.dictionary, spike_ui: mo.ui.dictionary) -> mo.Html:
    parts: list = [
        mo.md("### 描画設定"),
        mo.md(f"""
- 評価対象comp: {draw_ui["eval_comp"]}
- 描画関数: {draw_ui["draw_func"]}
"""),
    ]
    if "spike_orig" in spike_ui:
        parts.append(
            mo.md(
                f"- spike orig: {spike_ui['spike_orig']} / surr: {spike_ui['spike_surr']}"
            )
        )
    return mo.vstack(parts)


# ---------------------------------------------------------------------------
# Setting UI (集約)
# ---------------------------------------------------------------------------


def make_setting_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    current_type = str(base_ui["sim_current_type"].value)
    d: dict = {
        "surrogate_targets": _make_surrogate_targets_ui(base_ui),
        "sim": analysis_single.make_sim_ui(current_type),
        "run_sim": mo.ui.run_button(label="single 実行"),
    }
    sweep = analysis_sweep.make_sweep_ui(current_type)
    if sweep is not None:
        d["sweep"] = sweep
        d["run_sweep"] = mo.ui.run_button(label="sweep 実行")
    return mo.ui.dictionary(d)


def render_setting_ui(setting_ui: mo.ui.dictionary) -> mo.Html:
    parts: list = [
        mo.md(f"- {setting_ui['surrogate_targets']}"),
        analysis_single.render_sim_ui(setting_ui["sim"]),
        setting_ui["run_sim"],
    ]
    if "sweep" in setting_ui:
        parts.append(analysis_sweep.render_sweep(setting_ui["sweep"]))
        parts.append(setting_ui["run_sweep"])
    return mo.vstack(parts)


# ---------------------------------------------------------------------------
# Calc / Spike / View (集約)
# ---------------------------------------------------------------------------


def calc(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
    draw_ui: mo.ui.dictionary,
) -> dict | None:
    targets = setting_ui["surrogate_targets"].value
    if setting_ui["run_sim"].value:
        return analysis_single.calc_eval(base_ui, setting_ui["sim"], targets)
    if "run_sweep" in setting_ui and setting_ui["run_sweep"].value:
        return analysis_sweep.calc_sweep(base_ui, setting_ui["sweep"], targets, draw_ui)
    return None


def make_spike_ui(res: dict | None, draw_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    if res is None or "make_dm" not in res:
        return mo.ui.dictionary({})
    return analysis_single.make_spike_ui(res, draw_ui)


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------


def _get_model_info_figs(base_ui: mo.ui.dictionary) -> dict[str, Figure]:
    run_ids = cast(pd.DataFrame, base_ui["run_selector"].value)["run_id"].tolist()
    return {rid[:8]: RunInfo.get_run_info(rid).sindy_coef for rid in run_ids}


def _get_neurograph_fig(base_ui: mo.ui.dictionary) -> Figure:
    return view_neuron_graph(MCMODELS[str(base_ui["model_name"].value)])


def render_model_info(base_ui: mo.ui.dictionary) -> mo.Html:
    run_ids = cast(pd.DataFrame, base_ui["run_selector"].value)["run_id"].tolist()
    run_infos = [RunInfo.get_run_info(rid) for rid in run_ids]
    _model_name = str(base_ui["model_name"].value)
    return mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(f"run_id:{info.run_id[:8]}.. &nbsp;&nbsp;　{info.run_name}"),
                    mo.md(f"{info.equations[:40]}"),
                    mo.mpl.interactive(info.sindy_coef),
                ]
            )
            for info in run_infos
        ]
        + [
            mo.md(f"### NeuronGraph: `{_model_name}`"),
            mo.mpl.interactive(_get_neurograph_fig(base_ui)),
        ]
    )


def view(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
    res: dict | None,
    draw_ui: mo.ui.dictionary,
    spike_ui: mo.ui.dictionary,
) -> tuple[mo.Html, dict]:
    save_items: dict = {
        "neurograph": _get_neurograph_fig(base_ui),
        "current_preview": analysis_single.plot_current_preview(
            base_ui, setting_ui["sim"]
        ),
    }
    for k, v in _get_model_info_figs(base_ui).items():
        save_items[f"model_info_{k}"] = v

    if res is None:
        return mo.md("(結果なし)"), save_items
    if "make_dm" in res:
        _spike = spike_ui if "spike_orig" in spike_ui else None
        html, fig, dfs = analysis_single.view_result(draw_ui, res, _spike)
        save_items["waveform"] = fig
        save_items["metrics"] = dfs["metrics"]
        save_items["scalar_metrics"] = dfs["scalar_metrics"]
        return html, save_items
    html, fig = analysis_sweep.plot_sweep(res)
    save_items["sweep"] = fig
    return html, save_items


def plot_preview(base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary) -> mo.Html:
    fig = analysis_single.plot_current_preview(base_ui, setting_ui["sim"])
    return analysis_single.render_current_preview(fig)


# ---------------------------------------------------------------------------
# Save Panel
# ---------------------------------------------------------------------------


SaveItem = Figure | pd.DataFrame

SAVERS: dict[type, typing.Callable[[typing.Any, Path], None]] = {
    Figure: lambda o, p: o.savefig(p, dpi=300, bbox_inches="tight"),
    pd.DataFrame: lambda o, p: o.to_csv(p),
}


def _default_save_path(name: str, item: SaveItem | None) -> str:
    ext = ".csv" if isinstance(item, pd.DataFrame) else ".png"
    return f"_{name}{ext}"


def make_save_panel(save_items: dict[str, SaveItem | None]) -> mo.ui.dictionary:
    """save_items から各 name の path入力＋保存ボタンを生成。"""
    return mo.ui.dictionary(
        {
            name: mo.ui.dictionary(
                {
                    "path": mo.ui.text(
                        value=_default_save_path(name, item), label=name
                    ),
                    "save": mo.ui.run_button(label=f"{name} 保存"),
                }
            )
            for name, item in save_items.items()
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


def save(
    save_panel: mo.ui.dictionary, save_items: dict[str, SaveItem | None]
) -> mo.Html:
    msgs: list[mo.Html] = []
    for name, obj in save_items.items():
        ctrl = save_panel[name]
        if not ctrl["save"].value:
            continue
        if obj is None:
            msgs.append(mo.md(f"⚠️ {name}: 保存対象なし"))
            continue
        rel = str(ctrl["path"].value).strip()
        if not rel:
            msgs.append(mo.md(f"⚠️ {name}: パス未指定"))
            continue
        path = RESULT_DIR / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        SAVERS[type(obj)](obj, path)
        msgs.append(mo.md(f"✅ {name}: `{path.relative_to(REPO_ROOT)}`"))
    return mo.vstack(msgs) if msgs else mo.md("(未保存)")
