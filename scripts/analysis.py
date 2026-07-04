import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import analysis_single
import analysis_sweep
import marimo as mo
import matplotlib.pyplot as plt
import pandas as pd
from io_handler import RunInfo, get_runs_df, setup_mlflow
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


def make_base_ui() -> mo.ui.dictionary:
    runs_df = get_runs_df()
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


# ---------------------------------------------------------------------------
# Setting UI (集約)
# ---------------------------------------------------------------------------


def setup_mpl(matplotlib_style: str):
    style_dir = Path(__file__).resolve().parent / "conf" / "style"
    plt.style.use(style_dir / "base.mplstyle")
    plt.style.use(style_dir / f"{matplotlib_style}.mplstyle")


def _make_surrogate_targets_ui(base_ui: mo.ui.dictionary) -> mo.ui.multiselect:
    comp_names = MCMODELS[str(base_ui["model_name"].value)].names
    return mo.ui.multiselect(
        options=comp_names, value=[comp_names[0]], label="surrogate targets"
    )


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


# ---------------------------------------------------------------------------
# Calc
# ---------------------------------------------------------------------------


def calc(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
) -> dict | None:
    targets = setting_ui["surrogate_targets"].value
    if setting_ui["run_sim"].value:
        return analysis_single.calc_eval(base_ui, setting_ui["sim"], targets)
    if "run_sweep" in setting_ui and setting_ui["run_sweep"].value:
        return analysis_sweep.calc_sweep(base_ui, setting_ui["sweep"], targets)
    return None


# ---------------------------------------------------------------------------
# Draw setttings
# ---------------------------------------------------------------------------


def make_draw_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    comp_names = MCMODELS[str(base_ui["model_name"].value)].names
    return mo.ui.dictionary(
        {
            "eval_comp": mo.ui.dropdown(options=comp_names, value=comp_names[0], label="評価対象comp"),
            "draw_func": mo.ui.dropdown(options=DRAW_LIST, value=DRAW_LIST[0], label="描画関数"),
            "spike": mo.ui.dictionary(
                {
                    "orig": mo.ui.number(value=0, step=1, label="spike orig #"),
                    "surr": mo.ui.number(value=0, step=1, label="spike surr #"),
                }
            ),
        }
    )


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


# ---------------------------------------------------------------------------
# Result View
# ---------------------------------------------------------------------------


SaveItem = Figure | pd.DataFrame


@dataclass(frozen=True)
class SaveEntry:
    obj: SaveItem
    path: str  # default path (docs/slide/result 相対)


def _entry(name: str, obj: SaveItem, prefix: str) -> SaveEntry:
    ext = ".csv" if isinstance(obj, pd.DataFrame) else ".png"
    return SaveEntry(obj, f"_{prefix}_{name}{ext}")


def _fmt_current(base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary) -> str:
    ct = base_ui["sim_current_type"].value
    params = setting_ui["sim"]["current_params"].value or {}
    tail = "_".join(f"{k}{v}" for k, v in params.items())
    return f"{ct}_{tail}" if tail else ct


def _fmt_sweep(base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary) -> str:
    ct = base_ui["sim_current_type"].value
    s = setting_ui["sweep"]
    return (
        f"{ct}_sw{s['amp_start'].value:g}-{s['amp_stop'].value:g}"
        f"n{s['amp_steps'].value}"
    )


def view(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
    res: dict | None,
    draw_ui: mo.ui.dictionary,
) -> tuple[mo.Html, dict[str, SaveEntry]]:
    model = base_ui["model_name"].value
    current = _fmt_current(base_ui, setting_ui)
    single_prefix = f"{model}_{current}"

    entries: dict[str, SaveEntry] = {
        "neurograph": _entry("neurograph", _get_neurograph_fig(base_ui), model),
        "current_preview": _entry(
            "current_preview",
            analysis_single.plot_current_preview(base_ui, setting_ui["sim"]),
            current,
        ),
    }
    for rid, fig in _get_model_info_figs(base_ui).items():
        name = f"model_info_{rid}"
        entries[name] = _entry(name, fig, model)

    if res is None:
        return mo.md("(結果なし)"), entries
    if "make_dm" in res:
        html, fig, dfs = analysis_single.view_result(draw_ui, res)
        entries["waveform"] = _entry("waveform", fig, single_prefix)
        entries["metrics"] = _entry("metrics", dfs["metrics"], single_prefix)
        entries["scalar_metrics"] = _entry(
            "scalar_metrics", dfs["scalar_metrics"], single_prefix
        )
        return html, entries
    html, fig = analysis_sweep.plot_sweep(
        res,
        eval_comp_name=draw_ui["eval_comp"].value,
        metric_key=setting_ui["sweep"]["metric"].value,
    )
    entries["sweep"] = _entry("sweep", fig, f"{model}_{_fmt_sweep(base_ui, setting_ui)}")
    return html, entries


def plot_preview(base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary) -> mo.Html:
    fig = analysis_single.plot_current_preview(base_ui, setting_ui["sim"])
    return analysis_single.render_current_preview(fig)


# ---------------------------------------------------------------------------
# Save Panel
# ---------------------------------------------------------------------------


SAVERS: dict[type, typing.Callable[[typing.Any, Path], None]] = {
    Figure: lambda o, p: o.savefig(p, dpi=300, bbox_inches="tight"),
    pd.DataFrame: lambda o, p: o.to_csv(p),
}


def make_save_panel(entries: dict[str, SaveEntry]) -> mo.ui.dictionary:
    """SaveEntry から各 name の path入力＋保存ボタンを生成。"""
    return mo.ui.dictionary(
        {
            name: mo.ui.dictionary(
                {
                    "path": mo.ui.text(value=e.path, label=name),
                    "save": mo.ui.run_button(label=f"{name} 保存"),
                }
            )
            for name, e in entries.items()
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
    save_panel: mo.ui.dictionary, entries: dict[str, SaveEntry]
) -> mo.Html:
    msgs: list[mo.Html] = []
    for name, e in entries.items():
        ctrl = save_panel[name]
        if not ctrl["save"].value:
            continue
        rel = str(ctrl["path"].value).strip()
        if not rel:
            msgs.append(mo.md(f"⚠️ {name}: パス未指定"))
            continue
        path = RESULT_DIR / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        SAVERS[type(e.obj)](e.obj, path)
        msgs.append(mo.md(f"✅ {name}: `{path.relative_to(REPO_ROOT)}`"))
    return mo.vstack(msgs) if msgs else mo.md("(未保存)")
