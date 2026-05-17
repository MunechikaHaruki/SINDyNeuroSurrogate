from typing import Any, cast

import marimo as mo
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from io_handler import load_surrogate_model
from matplotlib.figure import Figure

from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model.model_dataset import CurrentConfig
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

    results: dict[str, list] = {"amplitudes": list(amplitudes), "original": []}
    for rid in run_ids:
        results[rid] = []

    for amp in amplitudes:
        u = CurrentConfig(
            iteration=int(duration / dt),
            silence_steps=int(silence_duration / dt),
            pipeline=CurrentConfig.build_pipeline(current_type, {**base_current_params, sweep_param: amp}),
        ).build()
        orig_ds = unified_simulator(dt=dt, u=u, net=net)

        first_m: dict | None = None
        for rid in run_ids:
            dm = DynamicMetrics(
                orig_ds,
                unified_simulator(
                    dt=dt,
                    u=u,
                    net=net.with_surrogates(
                        targets={net.nodes[comp_id].name},
                        make_surr=surrogates[rid].make_surr_comp,
                    ),
                    surrogate_model=surrogates[rid],
                ),
                comp_id,
                dt,
            )
            m = {**WaveformMetrics(dm).compute(), **SpikeMetrics(dm).compute()}
            if first_m is None:
                first_m = m
            results[rid].append(float(m.get(
                "surr_spike_count" if metric_key == "spike_count" else metric_key,
                float("nan"),
            )))

        results["original"].append(
            float(first_m.get(
                "orig_spike_count" if metric_key == "spike_count" else metric_key,
                float("nan"),
            )) if first_m else float("nan")
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


def make_sweep_ui(param_keys: list[str]) -> mo.ui.dictionary:
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



def run_and_plot(
    sweep_ui: mo.ui.dictionary,
    base_button: mo.ui.dictionary,
    param_button: mo.ui.dictionary,
    eval_ui: mo.ui.dictionary,
    run_ids: list[str],
) -> Figure:
    ds = cast(dict[str, Any], base_button["base_dataset"].value)
    current_type = str(base_button["current_type"].value)
    base_current_params = cast(dict, param_button["current_params"].value)
    comp_name = str(eval_ui["eval_comp"].value)

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
        target_comp_name=comp_name,
        current_type=current_type,
        base_current_params=base_current_params,
        sweep_param=str(sweep_ui["sweep_param"].value),
        metric_key=str(sweep_ui["metric"].value),
    )
    return plot_sweep(
        data,
        run_ids,
        str(sweep_ui["metric"].value),
        comp_name,
        run_labels,
    )
