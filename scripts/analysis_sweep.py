from typing import Any, cast

import marimo as mo
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from analysis_core import get_comp_names
from io_handler import load_surrogate_model
from matplotlib.figure import Figure

from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model.model_dataset import NeuronGraph
from neurosurrogate.model.registry_neuron import MCMODELS
from neurosurrogate.profiler.profiler_wave import DynamicMetrics

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


def _make_steady_u(amp: float, iteration: int, silence_steps: int) -> np.ndarray:
    from neurosurrogate.builder.registry_current import generate_steady

    u = np.zeros(iteration)
    generate_steady(amp)(u[silence_steps : iteration - silence_steps])
    return u


def _run_surr_sim(orig_ds, net, comp_id, u, dt, surrogate):
    surr_nodes = [
        surrogate.make_surr_comp(n.name) if i == comp_id else n
        for i, n in enumerate(net.nodes)
    ]
    surr_graph = NeuronGraph(nodes=surr_nodes, edges=net.edges, stim=net.stim)
    return unified_simulator(dt=dt, u=u, net=surr_graph, surrogate_model=surrogate)


def sweep_amplitude_metrics(
    run_ids: list[str],
    amplitudes: list[float],
    model_name: str,
    dt: float,
    silence_duration: float,
    duration: float,
    target_comp_name: str,
    metric_key: str = "spike_count",
) -> dict:
    net: NeuronGraph = MCMODELS[model_name]
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
        u = _make_steady_u(amp, iteration, silence_steps)
        orig_ds = unified_simulator(dt=dt, u=u, net=net)

        first_surr_ds = _run_surr_sim(
            orig_ds, net, comp_id, u, dt, surrogates[run_ids[0]]
        )
        def _all_metrics(orig, surr):
            m = DynamicMetrics(orig, surr, comp_id, dt)
            return {**m.waveform_metrics(), **m.spike_shape_metrics()}

        first_m = _all_metrics(orig_ds, first_surr_ds)
        results["original"].append(float(first_m.get(orig_key, float("nan"))))

        for rid in run_ids:
            surr_ds = _run_surr_sim(orig_ds, net, comp_id, u, dt, surrogates[rid])
            m = _all_metrics(orig_ds, surr_ds)
            results[rid].append(float(m.get(surr_key, float("nan"))))

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


def make_sweep_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    comp_names = get_comp_names(
        str(cast(dict, base_ui["base_dataset"].value)["model_name"])
    )
    return mo.ui.dictionary(
        {
            "amp_start": mo.ui.number(value=-5.0, step=1.0, label="amp_start"),
            "amp_stop": mo.ui.number(value=20.0, step=1.0, label="amp_stop"),
            "amp_steps": mo.ui.number(value=10, step=1, label="steps"),
            "metric": mo.ui.dropdown(options=_SWEEP_METRICS, value="spike_count"),
            "comp": mo.ui.dropdown(options=comp_names, value=comp_names[0]),
        }
    )


def render_sweep(sweep_ui: mo.ui.dictionary) -> mo.Html:
    return mo.md(f"""
    ### 振幅スイープ設定
    | | |
    |---|---|
    | amp start | {sweep_ui["amp_start"]} |
    | amp stop  | {sweep_ui["amp_stop"]}  |
    | steps     | {sweep_ui["amp_steps"]} |
    | metric    | {sweep_ui["metric"]} |
    | comp      | {sweep_ui["comp"]} |
    """)


def run_and_plot(
    sweep_ui: mo.ui.dictionary,
    base_ui: mo.ui.dictionary,
    run_ids: list[str],
) -> Figure:
    ds = cast(dict[str, Any], base_ui["base_dataset"].value)
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
        metric_key=str(sweep_ui["metric"].value),
    )
    return plot_sweep(
        data,
        run_ids,
        str(sweep_ui["metric"].value),
        str(sweep_ui["comp"].value),
        run_labels,
    )
