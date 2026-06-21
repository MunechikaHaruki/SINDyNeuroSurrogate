import inspect
import os
import typing
from functools import partial
from typing import Literal, cast

import marimo as mo
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from io_handler import TARGET_EXP, RunInfo, load_surrogate_model, setup_mlflow

from neurosurrogate.builder.registry_current import FUNC_MAP
from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model.model_dataset import CurrentConfig, DatasetConfig
from neurosurrogate.model.model_neurosindy import transform_gate
from neurosurrogate.model.registry_neuron import MCMODELS
from neurosurrogate.profiler.profiler_view import view_neuron_graph
from neurosurrogate.profiler.profiler_wave import (
    DynamicMetrics,
    SpikeMetrics,
    WaveformMetrics,
)
from neurosurrogate.profiler.registry_view import DRAW_MAP

CurrentList: list = ["train"] + list(FUNC_MAP.keys())
DRAW_LIST: list = list(DRAW_MAP.keys())
MplStyle = Literal["paper", "presentation"]
MCNameList = list(MCMODELS.keys())

setup_mlflow()


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
            "plt_style": mo.ui.radio(options=plt_options, value=plt_options[0]),
            "current_type": mo.ui.dropdown(CurrentList, value="steady"),
            "base_dataset": mo.ui.dictionary(
                {
                    "dt": mo.ui.number(value=0.01, step=0.001, label="dt"),
                    "silence_duration": mo.ui.number(
                        value=80, step=1, label="silence_duration"
                    ),
                    "duration": mo.ui.number(value=800, step=100, label="duration"),
                    "model_name": mo.ui.dropdown(
                        options=list(MCMODELS.keys()),
                        label="model_name",
                        value="hh",
                    ),
                }
            ),
            "run_selector": mo.ui.table(
                pd.DataFrame(runs_df[["tags.mlflow.runName", "run_id"]]),
                label="比較・解析したいRunを複数選択",
                selection="multi",
                initial_selection=[0],
            ),
        }
    )


def render_base(base_ui: mo.ui.dictionary) -> mo.Html:
    return mo.md(f"""
    ### MLflow データ解析
    - CurrentType: {base_ui["current_type"]}
    - matplotlib rendering setting: {base_ui["plt_style"]}
    - baseDatasetUI: {base_ui["base_dataset"]}
    {base_ui["run_selector"]}
    """)


# ---------------------------------------------------------------------------
# Param UI
# ---------------------------------------------------------------------------


def setup_mpl(matplotlib_style: str):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    STYLE_DIR = os.path.join(CURRENT_DIR, "./conf/style")
    plt.style.use(os.path.join(STYLE_DIR, "./base.mplstyle"))
    plt.style.use(os.path.join(STYLE_DIR, f"./{matplotlib_style}.mplstyle"))


def _make_ui_element(name: str, annotation: type, default):
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


def make_param_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    import analysis_sweep

    model_name = str(cast(dict, base_ui["base_dataset"].value)["model_name"])
    comp_names = MCMODELS[model_name].names
    run_ids = cast(pd.DataFrame, base_ui["run_selector"].value)["run_id"].tolist()
    current_type = str(base_ui["current_type"].value)

    if current_type == "train":
        current_params_ui: mo.ui.dictionary = mo.ui.dictionary({})
        param_keys: list[str] = []
    else:
        current_sig = inspect.signature(FUNC_MAP[current_type])
        current_params_ui = mo.ui.dictionary(
            {
                name: _make_ui_element(
                    name,
                    param.annotation,
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else 0,
                )
                for name, param in current_sig.parameters.items()
            }
        )
        param_keys = list(current_sig.parameters.keys())

    return mo.ui.dictionary(
        {
            "current_params": current_params_ui,
            "surrogate_targets": mo.ui.multiselect(
                options=comp_names, value=[comp_names[0]]
            ),
            "run_id": mo.ui.dropdown(options=run_ids, value=run_ids[0]),
            "sweep_ui": analysis_sweep.make_sweep_ui(param_keys),
        }
    )


def render_param(param_ui: mo.ui.dictionary) -> mo.Html:
    sweep = param_ui["sweep_ui"]
    return mo.md(f"""
    ### パラメタ設定
    - currentui: {param_ui["current_params"]}
    - surrogate target: {param_ui["surrogate_targets"]}
    - run id: {param_ui["run_id"]}
    ### 振幅スイープ設定
    | | |
    |---|---|
    | sweep param | {sweep["sweep_param"]} |
    | amp start | {sweep["amp_start"]} |
    | amp stop  | {sweep["amp_stop"]}  |
    | steps     | {sweep["amp_steps"]} |
    | metric    | {sweep["metric"]} |
    """)


# ---------------------------------------------------------------------------
# Eval UI
# ---------------------------------------------------------------------------


def make_eval_ui(param_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    comp_options = cast(list[str], param_ui["surrogate_targets"].value)
    return mo.ui.dictionary(
        {
            "eval_comp": mo.ui.dropdown(options=comp_options, value=comp_options[0]),
            "draw_func": mo.ui.dropdown(options=DRAW_LIST, value=DRAW_LIST[0]),
        }
    )


def render_eval(eval_ui: mo.ui.dictionary) -> mo.Html:
    return mo.md(
        f"評価対象のComp:{eval_ui['eval_comp']},描画関数:{eval_ui['draw_func']}"
    )


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------


def render_model_info(base_ui: mo.ui.dictionary) -> mo.Html:
    run_ids = cast(pd.DataFrame, base_ui["run_selector"].value)["run_id"].tolist()
    run_infos = [RunInfo.get_run_info(rid) for rid in run_ids]
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
    )


def render_neurograph(base_ui: mo.ui.dictionary) -> mo.Html:
    _model_name = base_ui["base_dataset"].value["model_name"]
    return mo.vstack(
        [
            mo.md(f"### NeuronGraph: `{_model_name}`"),
            mo.mpl.interactive(view_neuron_graph(MCMODELS[_model_name])),
        ]
    )


def _build_dataset_cfg(
    current_type: str,
    run_id: str,
    current_params: dict | None,
    base_dataset_params: dict,
) -> DatasetConfig:
    if current_type == "train":
        return RunInfo.get_run_info(run_id).dataset
    assert current_params is not None
    return DatasetConfig.build_dataset(
        **base_dataset_params,
        pipeline=CurrentConfig.build_pipeline(current_type, current_params),
    )


def to_eval_params(
    base_button: mo.ui.dictionary,
    param_button: mo.ui.dictionary,
) -> tuple[DatasetConfig, str, list[str]]:
    current_type = str(base_button["current_type"].value)
    run_id = str(param_button["run_id"].value)
    current_params_val = param_button["current_params"].value
    current_params = current_params_val if current_params_val else None
    base_dataset_params = cast(dict, base_button["base_dataset"].value)
    surrogate_targets = cast(list[str], param_button["surrogate_targets"].value)
    dataset_cfg = _build_dataset_cfg(
        current_type, run_id, current_params, base_dataset_params
    )
    return dataset_cfg, run_id, surrogate_targets


def build_eval_result(
    dataset_cfg: DatasetConfig,
    run_id: str,
    surrogate_targets: list[str],
) -> dict:
    original_graph = dataset_cfg.net
    surrogate_model = load_surrogate_model(run_id)
    u = dataset_cfg.current.build()
    original_ds = unified_simulator(dt=dataset_cfg.dt, u=u, net=original_graph)

    surr_ds = unified_simulator(
        dt=dataset_cfg.dt,
        u=u,
        net=original_graph.with_surrogates(
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
    }


def make_spike_ui(result: dict, eval_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    target_comp_id = result["name_to_idx"](eval_ui["eval_comp"].value)
    dm = DynamicMetrics(
        result["original_ds"], result["surr_ds"], target_comp_id, result["dt"]
    )
    n_orig, n_surr = SpikeMetrics(dm).n_spikes
    max_n = max(n_orig, n_surr)
    options: dict = {"median": "median"} | {str(i): i for i in range(max_n)}
    return mo.ui.dictionary(
        {
            "spike": mo.ui.dropdown(
                options=options,
                value="median",
                label=f"spike # (orig: {n_orig}, surr: {n_surr})",
            )
        }
    )


def render_spike(spike_ui: mo.ui.dictionary) -> mo.Html:
    return mo.md(f"スパイク選択: {spike_ui['spike']}")


def view_result(
    eval_ui: mo.ui.dictionary,
    result: dict,
    spike_ui: mo.ui.dictionary | None = None,
) -> mo.Html:
    target_comp_id = result["name_to_idx"](eval_ui["eval_comp"].value)
    dm = DynamicMetrics(
        result["original_ds"], result["surr_ds"], target_comp_id, result["dt"]
    )
    pre = result["get_preprocessed"](target_comp_id)

    raw_spike = spike_ui["spike"].value if spike_ui is not None else "median"
    spike: int | Literal["median"] = (
        "median" if raw_spike == "median" else int(raw_spike)
    )

    def _stat_cards(d: dict) -> mo.Html:
        return mo.hstack(
            [
                mo.stat(label=k, value=f"{v:.4f}" if isinstance(v, float) else str(v))
                for k, v in d.items()
            ],
            wrap=True,
        )

    sm = SpikeMetrics(dm)
    wm = WaveformMetrics(dm)
    return mo.vstack(
        [
            mo.md("#### 波形・発火パターン指標（orig / surr / orig-surr）"),
            wm.to_df(),
            mo.md("#### 波形誤差スカラー"),
            _stat_cards(wm.compute()),
            mo.md("#### スパイク波形相関（spike_shape_corr）"),
            _stat_cards(sm.compute()),
            mo.md(f"#### AP・ISI 指標（orig / surr / orig-surr） — spike: {spike}"),
            sm.to_df(spike=spike),
            mo.mpl.interactive(
                DRAW_MAP[eval_ui["draw_func"].value](
                    result["original_ds"],
                    result["surr_ds"],
                    pre,
                    target_comp_id,
                )
            ),
        ]
    )
