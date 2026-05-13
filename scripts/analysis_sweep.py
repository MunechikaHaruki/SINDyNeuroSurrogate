from dataclasses import dataclass
from typing import Literal

import marimo as mo
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import yaml
from analysis import BaseUI
from io_handler import load_surrogate_model

from neurosurrogate.builder.build_current import (
    FUNC_MAP,
)
from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model.model_dataset import NeuronGraph, Node
from neurosurrogate.model.model_neuron import MCMODELS
from neurosurrogate.profiler.draw_registry import DRAW_MAP
from neurosurrogate.profiler.profiler_view import view_model
from neurosurrogate.profiler.profiler_wave import calc_dynamic_metrics

CurrentList: list = ["train"] + list(FUNC_MAP.keys())
DRAW_LIST: list = list(DRAW_MAP.keys())
MplStyle = Literal["paper", "presentation"]
MCNameList = list(MCMODELS.keys())

_SWEEP_METRICS = [
    "spike_count",
    "rmse",
    "mae",
    "latency_error",
    "periodicity_gap",
    "spike_count_diff",
    "median_amp_error",
]

get_comp_names = lambda base_btn: (
    MCMODELS[base_btn.base_dataset_ui.value["model_name"]].names
)

CurrentList: list = ["train"] + list(FUNC_MAP.keys())
DRAW_LIST: list = list(DRAW_MAP.keys())
MplStyle = Literal["paper", "presentation"]
MCNameList = list(MCMODELS.keys())

_SWEEP_METRICS = [
    "spike_count",
    "rmse",
    "mae",
    "latency_error",
    "periodicity_gap",
    "spike_count_diff",
    "median_amp_error",
]


def get_run_info(run_id: str) -> dict:
    client = mlflow.MlflowClient()

    def load_yaml(run_id: str, filename: str) -> dict:
        return yaml.safe_load(mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}"))

    def load_text(run_id: str, filename: str) -> str:
        return mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}")

    view_cfg = load_yaml(run_id, "view.json")

    return {
        "sindy_coef": view_model(**view_cfg),
        "dataset": load_yaml(run_id, "dataset.yaml"),
        "runName": client.get_run(run_id).data.tags["mlflow.runName"],
        "run_id": run_id,
        "equations": load_text(run_id, "equations.txt"),
    }


# ── sweep ─────────────────────────────────────────────────────────────────────


def _make_steady_u(amp: float, iteration: int, silence_steps: int) -> np.ndarray:
    from neurosurrogate.builder.build_current import generate_steady

    u = np.zeros(iteration)
    generate_steady(amp)(u[silence_steps : iteration - silence_steps])
    return u


def _run_surr_sim(orig_ds, net, comp_id, u, dt, surrogate):
    """1コンパートメントをsurrに差し替えてシミュレーション。"""
    surr_nodes = [
        Node(n.name, "surr") if i == comp_id else n for i, n in enumerate(net.nodes)
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
    """
    定常電流の振幅をスイープし、各 run_id のサロゲートとオリジナルの
    metric を収集する。

    Returns
    -------
    dict with keys: "amplitudes", "original", <run_id>, ...
    """
    name_to_idx = MCMODELS[model_name].name_to_idx
    comp_id = name_to_idx(target_comp_name)
    net: NeuronGraph = MCMODELS[model_name]

    surrogates = {rid: load_surrogate_model(rid) for rid in run_ids}

    iteration = int(duration / dt)
    silence_steps = int(silence_duration / dt)

    # orig/surr どちらの列を取るか
    orig_key = "orig_spike_count" if metric_key == "spike_count" else metric_key
    surr_key = "surr_spike_count" if metric_key == "spike_count" else metric_key

    results: dict[str, list] = {"amplitudes": list(amplitudes), "original": []}
    for rid in run_ids:
        results[rid] = []

    for amp in amplitudes:
        u = _make_steady_u(amp, iteration, silence_steps)
        orig_ds = unified_simulator(dt=dt, u=u, net=net)

        # オリジナルのメトリクスは最初のサロゲートとの比較から取得（orig_ 列は共通）
        first_surr_ds = _run_surr_sim(
            orig_ds, net, comp_id, u, dt, surrogates[run_ids[0]]
        )
        first_m = calc_dynamic_metrics(orig_ds, first_surr_ds, comp_id=comp_id, dt=dt)
        results["original"].append(float(first_m.get(orig_key, float("nan"))))

        for rid in run_ids:
            surr_ds = _run_surr_sim(orig_ds, net, comp_id, u, dt, surrogates[rid])
            m = calc_dynamic_metrics(orig_ds, surr_ds, comp_id=comp_id, dt=dt)
            results[rid].append(float(m.get(surr_key, float("nan"))))

    return results


def plot_sweep(
    data: dict,
    run_ids: list[str],
    metric_key: str,
    comp_name: str,
    run_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """sweep_amplitude_metrics の結果を figure に描画して返す。"""
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


@dataclass
class SweepUI:
    amp_start: mo.ui.number
    amp_stop: mo.ui.number
    amp_steps: mo.ui.number
    metric_dropdown: mo.ui.dropdown
    comp_dropdown: mo.ui.dropdown

    def render(self) -> mo.Html:
        return mo.md(f"""
        ### 振幅スイープ設定
        | | |
        |---|---|
        | amp start | {self.amp_start} |
        | amp stop  | {self.amp_stop}  |
        | steps     | {self.amp_steps} |
        | metric    | {self.metric_dropdown} |
        | comp      | {self.comp_dropdown} |
        """)

    @property
    def amplitudes(self) -> list[float]:
        return list(
            np.linspace(
                self.amp_start.value, self.amp_stop.value, int(self.amp_steps.value)
            )
        )

    @staticmethod
    def build(base_btn: BaseUI):
        comp_names = get_comp_names(base_btn)
        return SweepUI(
            amp_start=mo.ui.number(value=-5.0, step=1.0, label="amp_start"),
            amp_stop=mo.ui.number(value=20.0, step=1.0, label="amp_stop"),
            amp_steps=mo.ui.number(value=10, step=1, label="steps"),
            metric_dropdown=mo.ui.dropdown(options=_SWEEP_METRICS, value="spike_count"),
            comp_dropdown=mo.ui.dropdown(options=comp_names, value=comp_names[0]),
        )

    def run_and_plot(self, base_btn: "BaseUI", run_ids: list[str]) -> plt.Figure:
        ds = base_btn.base_dataset_ui.value
        client = mlflow.MlflowClient()
        run_labels = {
            rid: client.get_run(rid).data.tags.get("mlflow.runName", rid[:6])
            for rid in run_ids
        }
        data = sweep_amplitude_metrics(
            run_ids=run_ids,
            amplitudes=self.amplitudes,
            model_name=ds["model_name"],
            dt=ds["dt"],
            silence_duration=ds["silence_duration"],
            duration=ds["duration"],
            target_comp_name=self.comp_dropdown.value,
            metric_key=self.metric_dropdown.value,
        )
        return plot_sweep(
            data,
            run_ids,
            self.metric_dropdown.value,
            self.comp_dropdown.value,
            run_labels,
        )
