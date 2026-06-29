import inspect
import typing
from functools import partial
from pathlib import Path
from typing import Literal, cast

import analysis_sweep
import marimo as mo
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from io_handler import TARGET_EXP, RunInfo, load_surrogate_model, setup_mlflow
from matplotlib.figure import Figure

from neurosurrogate.builder.registry_current import FUNC_MAP
from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model.model_dataset import CurrentConfig, DatasetConfig
from neurosurrogate.model.model_neurosindy import transform_gate
from neurosurrogate.model.registry_neuron import MCMODELS
from neurosurrogate.profiler.profiler_view import view_neuron_graph
from neurosurrogate.profiler.profiler_wave import (
    DynamicMetrics,
    n_spikes,
    spike_features_df,
    spike_shape_corr,
    waveform_summary,
    waveform_summary_df,
)
from neurosurrogate.profiler.registry_view import DRAW_MAP

CurrentList: list = list(FUNC_MAP.keys())
DRAW_LIST: list = list(DRAW_MAP.keys())
MplStyle = Literal["paper", "presentation"]
MCNameList = list(MCMODELS.keys())

setup_mlflow()

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "docs" / "slide" / "result"


# ---------------------------------------------------------------------------
# Save Panel
# ---------------------------------------------------------------------------


def make_save_panel(defaults: dict[str, str]) -> mo.ui.dictionary:
    """defaults = {name: default_path}. 各nameごとに path入力＋保存ボタンを生成。"""
    return mo.ui.dictionary(
        {
            name: mo.ui.dictionary(
                {
                    "path": mo.ui.text(value=default, label=name),
                    "save": mo.ui.run_button(label=f"{name} 保存"),
                }
            )
            for name, default in defaults.items()
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


SaveItem = Figure | pd.DataFrame

SAVERS: dict[type, typing.Callable[[typing.Any, Path], None]] = {
    Figure: lambda o, p: o.savefig(p, dpi=300, bbox_inches="tight"),
    pd.DataFrame: lambda o, p: o.to_csv(p),
}


def save_panel_items(panel: mo.ui.dictionary, items: dict[str, SaveItem]) -> mo.Html:
    msgs: list[mo.Html] = []
    for name, obj in items.items():
        ctrl = panel[name]
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


# ---------------------------------------------------------------------------
# Base UI
# ---------------------------------------------------------------------------


def get_runs_df():
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
            "sweep_run_selector": mo.ui.table(
                pd.DataFrame(runs_df[["tags.mlflow.runName", "run_id"]]),
                label="sweep Run 選択（複数可）",
                selection="multi",
                initial_selection=[0],
            ),
            "eval_run_selector": mo.ui.table(
                pd.DataFrame(runs_df[["tags.mlflow.runName", "run_id"]]),
                label="単一評価 Run 選択",
                selection="single",
                initial_selection=[0],
            ),
        }
    )


# ---------------------------------------------------------------------------
# Param UI
# ---------------------------------------------------------------------------


def setup_mpl(matplotlib_style: str):
    style_dir = Path(__file__).resolve().parent / "conf" / "style"
    plt.style.use(style_dir / "base.mplstyle")
    plt.style.use(style_dir / f"{matplotlib_style}.mplstyle")


def _make_ui_element(name: str, annotation, default):
    if typing.get_origin(annotation) is Literal:
        options = list(typing.get_args(annotation))
        return mo.ui.dropdown(
            options=options,
            value=default if default in options else options[0],
            label=name,
        )
    if annotation is int:
        return mo.ui.number(value=int(default), step=1, label=name)
    elif annotation is float:
        return mo.ui.number(value=float(default), step=0.1, label=name)
    elif annotation is bool:
        return mo.ui.checkbox(value=bool(default), label=name)
    elif annotation is list:
        return mo.ui.array([mo.ui.number(value=0.0, step=0.1)], label=name)
    else:
        raise NotImplementedError(f"{name}: {annotation} は未対応の型です")


def make_sim_ui(base_ui: mo.ui.dictionary, current_type: str) -> mo.ui.dictionary:
    current_params_ui = mo.ui.dictionary(
        {
            name: _make_ui_element(
                name,
                param.annotation,
                0 if param.default is inspect.Parameter.empty else param.default,
            )
            for name, param in inspect.signature(
                FUNC_MAP[current_type]
            ).parameters.items()
        }
    )
    return mo.ui.dictionary({"current_params": current_params_ui})


def plot_current_preview(
    base_ui: mo.ui.dictionary, sim_ui: mo.ui.dictionary
) -> Figure | None:
    """sim_ui の current_params から電流波形を構築してプレビュー描画。
    構築失敗時は None。"""
    current_type = str(base_ui["sim_current_type"].value)
    dt = float(base_ui["dt"].value)
    params = sim_ui["current_params"].value or {}
    try:
        i_ext = CurrentConfig(
            pipeline=CurrentConfig.build_pipeline(current_type, params)
        ).build(dt)
    except Exception as e:  # noqa: BLE001
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.text(0.5, 0.5, f"build失敗: {e}", ha="center", va="center", fontsize=8)
        ax.axis("off")
        return fig
    t = np.arange(len(i_ext)) * dt
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(t, i_ext, lw=0.8)
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("I_ext [μA/cm²]")
    ax.set_title(f"{current_type} preview")
    fig.tight_layout()
    return fig


def render_sim_ui(sim_ui: mo.ui.dictionary) -> mo.Html:
    return mo.vstack(
        [
            mo.md("### シミュレーション設定"),
            mo.md(f"""
- current params: {sim_ui["current_params"]}
"""),
        ]
    )


def render_current_preview(fig: Figure | None) -> mo.Html:
    if fig is None:
        return mo.md("（プレビューなし）")
    return mo.vstack(
        [
            mo.md("### 電流プレビュー"),
            mo.mpl.interactive(fig),
        ]
    )


def make_combined_ui(base_ui: mo.ui.dictionary) -> dict:
    comp_names = MCMODELS[str(base_ui["model_name"].value)].names
    return {
        "surrogate_targets": mo.ui.multiselect(
            options=comp_names, value=[comp_names[0]]
        ),
        "sim": make_sim_ui(base_ui, str(base_ui["sim_current_type"].value)),
        "sweep": analysis_sweep.make_sweep_ui(str(base_ui["sim_current_type"].value)),
    }


def calc_sweep(
    base_button: mo.ui.dictionary,
    sweep_ui: mo.ui.dictionary | None,
    surrogate_targets: list[str],
    draw_ui: mo.ui.dictionary,
) -> dict:
    if sweep_ui is None:
        return {}
    return analysis_sweep.calc_sweep(base_button, sweep_ui, surrogate_targets, draw_ui)


def plot_sweep(sweep_result: dict) -> tuple[mo.Html, Figure | None]:
    return analysis_sweep.plot_sweep(sweep_result)


def render_combined_ui(combined_ui: dict) -> mo.Html:
    sweep_ui = combined_ui["sweep"]
    sweep_html = (
        analysis_sweep.render_sweep(sweep_ui) if sweep_ui is not None else mo.md("")
    )
    return mo.vstack(
        [
            mo.md(f"- surrogate targets: {combined_ui['surrogate_targets']}"),
            mo.hstack(
                [render_sim_ui(combined_ui["sim"]), sweep_html],
                align="start",
            ),
        ]
    )


def make_draw_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    comp_names = MCMODELS[str(base_ui["model_name"].value)].names
    return mo.ui.dictionary(
        {
            "eval_comp": mo.ui.dropdown(options=comp_names, value=comp_names[0]),
            "draw_func": mo.ui.dropdown(options=DRAW_LIST, value=DRAW_LIST[0]),
        }
    )


def render_draw_ui(draw_ui: mo.ui.dictionary) -> mo.Html:
    return mo.vstack(
        [
            mo.md("### 描画設定"),
            mo.md(f"""
- 評価対象comp: {draw_ui["eval_comp"]}
- 描画関数: {draw_ui["draw_func"]}
"""),
        ]
    )


def render_spike_ui(spike_ui: mo.ui.dictionary) -> mo.Html:
    return mo.md(f"""
- spike orig: {spike_ui["spike_orig"]} / surr: {spike_ui["spike_surr"]}
""")


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------


def get_model_info_figs(base_ui: mo.ui.dictionary) -> dict[str, Figure]:
    run_ids = cast(pd.DataFrame, base_ui["sweep_run_selector"].value)["run_id"].tolist()
    return {rid[:8]: RunInfo.get_run_info(rid).sindy_coef for rid in run_ids}


def get_neurograph_fig(base_ui: mo.ui.dictionary) -> Figure:
    return view_neuron_graph(MCMODELS[str(base_ui["model_name"].value)])


def render_model_info(base_ui: mo.ui.dictionary) -> mo.Html:
    run_ids = cast(pd.DataFrame, base_ui["sweep_run_selector"].value)["run_id"].tolist()
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
            mo.mpl.interactive(get_neurograph_fig(base_ui)),
        ]
    )


# ---------------------------------------------------------------------------
# Calc Result
# ---------------------------------------------------------------------------


def _parse_eval_button(
    base_button: mo.ui.dictionary,
    sim_ui: mo.ui.dictionary,
    surrogate_targets: list[str],
) -> tuple[DatasetConfig, str]:
    run_id = str(
        cast(pd.DataFrame, base_button["eval_run_selector"].value)["run_id"].iloc[0]
    )
    current_type = str(base_button["sim_current_type"].value)
    current_params = sim_ui["current_params"].value or {}
    dataset_cfg = DatasetConfig.build_dataset(
        model_name=str(base_button["model_name"].value),
        dt=float(base_button["dt"].value),
        pipeline=CurrentConfig.build_pipeline(current_type, current_params),
    )
    return dataset_cfg, run_id


def calc_eval(
    base_button: mo.ui.dictionary,
    sim_ui: mo.ui.dictionary,
    surrogate_targets: list[str],
) -> dict:
    dataset_cfg, run_id = _parse_eval_button(base_button, sim_ui, surrogate_targets)

    surrogate_model = load_surrogate_model(run_id)
    original_ds = unified_simulator(dataset_cfg)
    surr_ds = unified_simulator(
        dataset_cfg.with_surrogates(
            targets=set(surrogate_targets),
            make_surr=surrogate_model.make_surr_comp,
        ),
        surrogate_model=surrogate_model,
    )

    return {
        "original_ds": original_ds,
        "surr_ds": surr_ds,
        "dt": dataset_cfg.dt,
        "get_preprocessed": partial(
            transform_gate, surrogate_model.preprocessor, original_ds
        ),
        "name_to_idx": MCMODELS[dataset_cfg.model_name].name_to_idx,
        "make_dm": lambda comp_id: DynamicMetrics(
            original_ds, surr_ds, comp_id, dataset_cfg.dt
        ),
    }


# ---------------------------------------------------------------------------
# Select Spike
# ---------------------------------------------------------------------------


def make_spike_ui(result: dict, draw_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    dm = result["make_dm"](result["name_to_idx"](draw_ui["eval_comp"].value))
    n_orig, n_surr = n_spikes(dm)
    orig_options: dict = {str(i): i for i in range(n_orig)}
    surr_options: dict = {str(i): i for i in range(n_surr)}
    return mo.ui.dictionary(
        {
            "spike_orig": mo.ui.dropdown(
                options=orig_options,
                value="0" if n_orig > 0 else None,
                label=f"orig spike # (n={n_orig})",
            ),
            "spike_surr": mo.ui.dropdown(
                options=surr_options,
                value="0" if n_surr > 0 else None,
                label=f"surr spike # (n={n_surr})",
            ),
        }
    )


# ---------------------------------------------------------------------------
# View Result
# ---------------------------------------------------------------------------


def _spike_idx(spike_ui: mo.ui.dictionary | None, key: str) -> int:
    if spike_ui is None:
        return 0
    v = spike_ui[key].value
    return int(v) if v is not None else 0


def _stat_cards(d: dict) -> mo.Html:
    return mo.hstack(
        [
            mo.stat(label=k, value=f"{v:.4f}" if isinstance(v, float) else str(v))
            for k, v in d.items()
        ],
        wrap=True,
    )


def view_result(
    draw_ui: mo.ui.dictionary,
    result: dict,
    spike_ui: mo.ui.dictionary | None = None,
) -> tuple[mo.Html, Figure, dict[str, pd.DataFrame]]:
    target_comp_id = result["name_to_idx"](draw_ui["eval_comp"].value)
    dm = result["make_dm"](target_comp_id)
    spike_orig = _spike_idx(spike_ui, "spike_orig")
    spike_surr = _spike_idx(spike_ui, "spike_surr")

    wf_summary = waveform_summary(dm)
    spike_corr = spike_shape_corr(dm)
    df_waveform = waveform_summary_df(dm)
    df_spike = spike_features_df(dm, spike_orig=spike_orig, spike_surr=spike_surr)
    df_spike.index.name = "metric"
    df_metrics = pd.concat([df_waveform, df_spike])
    df_scalar = pd.DataFrame(
        {**wf_summary, **spike_corr}.items(),
        columns=["metric", "value"],
    ).set_index("metric")

    fig = DRAW_MAP[draw_ui["draw_func"].value](
        result["original_ds"],
        result["surr_ds"],
        result["get_preprocessed"](target_comp_id),
        target_comp_id,
    )

    html = mo.vstack(
        [
            mo.md(
                f"#### 動的指標（orig / surr / orig-surr） — spike orig: {spike_orig} / surr: {spike_surr}"
            ),
            df_metrics,
            mo.md("#### 波形誤差スカラー"),
            _stat_cards(wf_summary),
            mo.md("#### スパイク波形相関（spike_shape_corr）"),
            _stat_cards(spike_corr),
            mo.mpl.interactive(fig),
        ]
    )
    return (
        html,
        fig,
        {
            "metrics": df_metrics,
            "scalar_metrics": df_scalar,
        },
    )
