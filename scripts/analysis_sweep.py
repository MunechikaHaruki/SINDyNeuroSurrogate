import inspect
from typing import Any, cast

import marimo as mo
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from io_handler import load_surrogate_model
from matplotlib.figure import Figure

from neurosurrogate.builder.registry_current import FUNC_MAP
from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model.model_dataset import CurrentConfig
from neurosurrogate.model.registry_neuron import MCMODELS
from neurosurrogate.profiler.profiler_wave import (
    DynamicMetrics,
    SpikeMetrics,
    WaveformMetrics,
)

# origの絶対値が意味を持つ指標（compute()がorig_{key}/surr_{key}を返す）
_METRIC_SPLIT_KEYS: list[str] = ["spike_count"]
# origとsurrの比較値そのものが指標（surr列のみ格納）
_SWEEP_METRICS_COMPARISON: list[str] = [
    "rmse",
    "mae",
    "latency_error",
    "periodicity_gap",
    "spike_count_diff",
    "median_amp_error",
]
_SWEEP_METRICS = list(_METRIC_SPLIT_KEYS) + _SWEEP_METRICS_COMPARISON


def sweep_amplitude_metrics(
    surrogates: dict,
    current_configs: dict[float, CurrentConfig],
    model_name: str,
    dt: float,
    target_comp_name: str,
    metric_key: str,
) -> pd.DataFrame:
    net = MCMODELS[model_name]
    comp_id = net.name_to_idx(target_comp_name)
    has_orig = metric_key in _METRIC_SPLIT_KEYS
    orig_key = f"orig_{metric_key}" if has_orig else metric_key
    surr_key = f"surr_{metric_key}" if has_orig else metric_key

    frames: list[pd.DataFrame] = []
    for amp, cfg in current_configs.items():
        u = cfg.build()
        orig_ds = unified_simulator(dt=dt, u=u, net=net)
        surr_ms: list[tuple[str, dict]] = []
        for rid, surrogate in surrogates.items():
            dm = DynamicMetrics(
                orig_ds,
                unified_simulator(
                    dt=dt,
                    u=u,
                    net=net.with_surrogates(
                        targets={net.nodes[comp_id].name},
                        make_surr=surrogate.make_surr_comp,
                    ),
                    surrogate_model=surrogate,
                ),
                comp_id,
                dt,
            )
            surr_ms.append(
                (rid, {**WaveformMetrics(dm).compute(), **SpikeMetrics(dm).compute()})
            )

        surr_vals = {rid: float(m.get(surr_key, float("nan"))) for rid, m in surr_ms}
        if has_orig:
            orig_val = (
                float(surr_ms[0][1].get(orig_key, float("nan")))
                if surr_ms
                else float("nan")
            )
            frames.append(
                pd.DataFrame(
                    [[amp, orig_val, *surr_vals.values()]],
                    columns=["amplitude", "original", *surr_vals.keys()],
                )
            )
        else:
            frames.append(
                pd.DataFrame(
                    [[amp, *surr_vals.values()]],
                    columns=["amplitude", *surr_vals.keys()],
                )
            )

    return pd.concat(frames, ignore_index=True)


def plot_sweep(
    data: pd.DataFrame,
    run_ids: list[str],
    metric_key: str,
    comp_name: str,
    run_labels: dict[str, str] | None = None,
) -> Figure:
    run_labels = run_labels or {}

    fig, ax = plt.subplots()
    if "original" in data.columns:
        ax.plot(data["amplitude"], data["original"], "k-o", label="Original", zorder=3)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, rid in enumerate(run_ids):
        label = run_labels.get(rid, f"Surr {rid[:6]}")
        ax.plot(
            data["amplitude"],
            data[rid],
            marker="s",
            linestyle="--",
            color=colors[idx % len(colors)],
            label=label,
        )

    ax.set_xlabel("Stimulus amplitude [μA/cm²]")
    ax.set_ylabel(metric_key)
    ax.set_title(f"Amplitude sweep — {metric_key} ({comp_name})")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sweep UI
# ---------------------------------------------------------------------------


def _resolve_param_keys(current_type: str) -> list[str]:
    if current_type == "train":
        return []
    return list(inspect.signature(FUNC_MAP[current_type]).parameters.keys())


def make_sweep_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    param_keys = _resolve_param_keys(str(base_ui["current_type"].value))
    sweep_param_ui = (
        mo.ui.dropdown(options=param_keys, value=param_keys[0])
        if param_keys
        else mo.ui.dropdown(options=["(none)"], value="(none)")
    )

    return mo.ui.dictionary(
        {
            "sweep_param": sweep_param_ui,
            "amp_start": mo.ui.number(value=-5.0, step=1.0, label="amp_start"),
            "amp_stop": mo.ui.number(value=20.0, step=1.0, label="amp_stop"),
            "amp_steps": mo.ui.number(value=10, step=1, label="steps"),
            "metric": mo.ui.dropdown(options=_SWEEP_METRICS, value="spike_count"),
        }
    )


def render_sweep(sweep_ui: mo.ui.dictionary) -> mo.Html:
    return mo.md(f"""
    ### 振幅スイープ設定
    | | |
    |---|---|
    | sweep param | {sweep_ui["sweep_param"]} |
    | amp start | {sweep_ui["amp_start"]} |
    | amp stop  | {sweep_ui["amp_stop"]}  |
    | steps     | {sweep_ui["amp_steps"]} |
    | metric    | {sweep_ui["metric"]} |
    """)


def run_and_plot(
    sweep_ui: mo.ui.dictionary,
    base_button: mo.ui.dictionary,
    param_button: mo.ui.dictionary,
    eval_ui: mo.ui.dictionary,
    run_ids: list[str],
) -> tuple[pd.DataFrame, Figure]:
    ds = cast(dict[str, Any], base_button["base_dataset"].value)
    dt = float(ds["dt"])
    comp_name = str(eval_ui["eval_comp"].value)
    base_current_params = cast(dict, param_button["current_params"].value)
    sweep_param = str(sweep_ui["sweep_param"].value)

    current_configs: dict[float, CurrentConfig] = {
        amp: CurrentConfig(
            iteration=int(float(ds["duration"]) / dt),
            silence_steps=int(float(ds["silence_duration"]) / dt),
            pipeline=CurrentConfig.build_pipeline(
                str(base_button["current_type"].value),
                {**base_current_params, sweep_param: amp},
            ),
        )
        for amp in np.linspace(
            float(cast(Any, sweep_ui["amp_start"]).value),
            float(cast(Any, sweep_ui["amp_stop"]).value),
            int(cast(Any, sweep_ui["amp_steps"]).value),
        )
    }

    surrogates = {rid: load_surrogate_model(rid) for rid in run_ids}
    client = mlflow.MlflowClient()
    metric_key = str(sweep_ui["metric"].value)
    run_labels = {
        rid: client.get_run(rid).data.tags.get("mlflow.runName", rid[:6])
        for rid in run_ids
    }
    data = sweep_amplitude_metrics(
        surrogates=surrogates,
        current_configs=current_configs,
        model_name=str(ds["model_name"]),
        dt=dt,
        target_comp_name=comp_name,
        metric_key=metric_key,
    )
    fig = plot_sweep(data, run_ids, metric_key, comp_name, run_labels)
    return data, fig
