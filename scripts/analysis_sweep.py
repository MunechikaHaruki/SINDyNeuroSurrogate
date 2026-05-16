from typing import Any, cast

import marimo as mo
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from io_handler import load_surrogate_model
from matplotlib.figure import Figure

from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model.registry_neuron import MCMODELS
from neurosurrogate.profiler.profiler_wave import (
    DynamicMetrics,
    SpikeMetrics,
    WaveformMetrics,
)

_SWEEP_METRICS = [
    "spike_count",
    "rmse",
    "mae",
    "latency_error",
    "periodicity_gap",
    "spike_count_diff",
    "median_amp_error",
]


# ---------------------------------------------------------------------------
# Domain logic
# ---------------------------------------------------------------------------


def _make_current_u(
    current_type: str,
    base_params: dict,
    sweep_param: str,
    amp: float,
    iteration: int,
    silence_steps: int,
) -> np.ndarray:
    from neurosurrogate.builder.registry_current import FUNC_MAP

    params = {**base_params, sweep_param: amp}
    u = np.zeros(iteration)
    FUNC_MAP[current_type](**params)(u[silence_steps : iteration - silence_steps])
    return u


def _run_surr_sim(net, comp_id: int, u, dt: float, surrogate):
    surr_graph = net.with_surrogates(
        targets={net.nodes[comp_id].name},
        make_surr=surrogate.make_surr_comp,
    )
    return unified_simulator(dt=dt, u=u, net=surr_graph, surrogate_model=surrogate)


def _all_metrics(orig, surr, comp_id: int, dt: float) -> dict:
    dm = DynamicMetrics(orig, surr, comp_id, dt)
    return {**WaveformMetrics(dm).compute(), **SpikeMetrics(dm).compute()}


def sweep_amplitude_metrics(
    run_ids: list[str],
    amplitudes: list[float],
    model_name: str,
    dt: float,
    silence_duration: float,
    duration: float,
    target_comp_name: str,
    current_type: str,
    base_current_params: dict,
    sweep_param: str,
    metric_key: str = "spike_count",
) -> dict:
    net = MCMODELS[model_name]
    comp_id = net.name_to_idx(target_comp_name)
    surrogates = {rid: load_surrogate_model(rid) for rid in run_ids}

    iteration = int(duration / dt)
    silence_steps = int(silence_duration / dt)

    orig_key = "orig_spike_count" if metric_key == "spike_count" else metric_key
    surr_key = "surr_spike_count" if metric_key == "spike_count" else metric_key

    results: dict[str, list] = {"amplitudes": list(amplitudes), "original": []}
    for rid in run_ids:
        results[rid] = []

    for amp in amplitudes:
        u = _make_current_u(current_type, base_current_params, sweep_param, amp, iteration, silence_steps)
        orig_ds = unified_simulator(dt=dt, u=u, net=net)

        first_m: dict | None = None
        for rid in run_ids:
            surr_ds = _run_surr_sim(net, comp_id, u, dt, surrogates[rid])
            m = _all_metrics(orig_ds, surr_ds, comp_id, dt)
            if first_m is None:
                first_m = m
            results[rid].append(float(m.get(surr_key, float("nan"))))

        results["original"].append(
            float(first_m.get(orig_key, float("nan"))) if first_m else float("nan")
        )

    return results


def plot_sweep(
    data: dict,
    run_ids: list[str],
    metric_key: str,
    comp_name: str,
    run_labels: dict[str, str] | None = None,
) -> Figure:
    amps = data["amplitudes"]
    run_labels = run_labels or {}

    fig, ax = plt.subplots()
    ax.plot(amps, data["original"], "k-o", label="Original", zorder=3)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, rid in enumerate(run_ids):
        label = run_labels.get(rid, f"Surr {rid[:6]}")
        ax.plot(
            amps,
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


def make_sweep_ui(
    param_button: mo.ui.dictionary,
    eval_ui: mo.ui.dictionary,
) -> mo.ui.dictionary:
    comp_options = cast(list[str], param_button["surrogate_targets"].value)
    comp_value = str(eval_ui["eval_comp"].value)
    if comp_value not in comp_options:
        comp_value = comp_options[0]

    current_params = cast(dict, param_button["current_params"].value)
    param_keys = list(current_params.keys())

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
            "comp": mo.ui.dropdown(options=comp_options, value=comp_value),
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
    | comp      | {sweep_ui["comp"]} |
    """)


def run_and_plot(
    sweep_ui: mo.ui.dictionary,
    base_button: mo.ui.dictionary,
    param_button: mo.ui.dictionary,
    run_ids: list[str],
) -> Figure:
    ds = cast(dict[str, Any], base_button["base_dataset"].value)
    current_type = str(base_button["current_type"].value)
    base_current_params = cast(dict, param_button["current_params"].value)

    client = mlflow.MlflowClient()
    run_labels = {
        rid: client.get_run(rid).data.tags.get("mlflow.runName", rid[:6])
        for rid in run_ids
    }
    amplitudes = list(
        np.linspace(
            float(cast(Any, sweep_ui["amp_start"]).value),
            float(cast(Any, sweep_ui["amp_stop"]).value),
            int(cast(Any, sweep_ui["amp_steps"]).value),
        )
    )
    data = sweep_amplitude_metrics(
        run_ids=run_ids,
        amplitudes=amplitudes,
        model_name=str(ds["model_name"]),
        dt=float(ds["dt"]),
        silence_duration=float(ds["silence_duration"]),
        duration=float(ds["duration"]),
        target_comp_name=str(sweep_ui["comp"].value),
        current_type=current_type,
        base_current_params=base_current_params,
        sweep_param=str(sweep_ui["sweep_param"].value),
        metric_key=str(sweep_ui["metric"].value),
    )
    return plot_sweep(
        data,
        run_ids,
        str(sweep_ui["metric"].value),
        str(sweep_ui["comp"].value),
        run_labels,
    )
