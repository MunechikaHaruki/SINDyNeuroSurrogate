import inspect
from collections.abc import Iterator
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
    DF_ROW_METRICS,
    SCALAR_METRICS,
    DynamicMetrics,
    extract_metric,
)

# ---------------------------------------------------------------------------
# Sweep UI
# ---------------------------------------------------------------------------


def make_sweep_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary:
    current_type = str(base_ui["current_type"].value)
    param_keys = (
        []
        if current_type == "train"
        else list(inspect.signature(FUNC_MAP[current_type]).parameters.keys())
    )
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
            "metric": mo.ui.dropdown(
                options=DF_ROW_METRICS + SCALAR_METRICS, value="spike_count"
            ),
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


# ---------------------------------------------------------------------------
# Run and Sweep
# ---------------------------------------------------------------------------


def _iter_amp_dms(
    surrogates: dict,
    current_configs: dict[float, CurrentConfig],
    model_name: str,
    dt: float,
    target_comp_name: str,
) -> Iterator[tuple[float, dict[str, DynamicMetrics]]]:
    """各 amp で {rid: DynamicMetrics} を yield。orig_ds は amp ごとに1回計算。"""
    net = MCMODELS[model_name]
    comp_id = net.name_to_idx(target_comp_name)
    for amp, cfg in current_configs.items():
        u = cfg.build()
        orig_ds = unified_simulator(dt=dt, u=u, net=net)
        dms: dict[str, DynamicMetrics] = {}
        for rid, surrogate in surrogates.items():
            surr_ds = unified_simulator(
                dt=dt,
                u=u,
                net=net.with_surrogates(
                    targets={net.nodes[comp_id].name},
                    make_surr=surrogate.make_surr_comp,
                ),
                surrogate_model=surrogate,
            )
            dms[rid] = DynamicMetrics(orig_ds, surr_ds, comp_id, dt)
        yield amp, dms


def _sweep_amplitude_metrics(
    surrogates: dict,
    current_configs: dict[float, CurrentConfig],
    model_name: str,
    dt: float,
    target_comp_name: str,
    metric_key: str,
) -> pd.DataFrame:
    """amp × rid を走査し metric DataFrame を構築。

    orig 列の有無は extract_metric の戻り値 (None かどうか) で決まる。
    """
    rows: list[dict] = []
    for amp, dms in _iter_amp_dms(
        surrogates, current_configs, model_name, dt, target_comp_name
    ):
        extracted = {rid: extract_metric(dm, metric_key) for rid, dm in dms.items()}
        # orig は rid 非依存 → 任意の 1 件から取得
        orig_val = next(iter(extracted.values()))[0]
        row: dict = {"amplitude": amp}
        if orig_val is not None:
            row["original"] = orig_val
        row.update({rid: surr for rid, (_, surr) in extracted.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_sweep(
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


def run_sweep(
    *,
    run_ids: list[str],
    model_name: str,
    dt: float,
    duration: float,
    silence_duration: float,
    comp_name: str,
    current_type: str,
    base_current_params: dict,
    sweep_param: str,
    amp_start: float,
    amp_stop: float,
    amp_steps: int,
    metric_key: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """純粋計算層: metric DataFrame と run_label dict を返す。plot しない。"""
    current_configs: dict[float, CurrentConfig] = {
        amp: CurrentConfig(
            iteration=int(duration / dt),
            silence_steps=int(silence_duration / dt),
            pipeline=CurrentConfig.build_pipeline(
                current_type, {**base_current_params, sweep_param: amp}
            ),
        )
        for amp in np.linspace(amp_start, amp_stop, amp_steps)
    }
    surrogates = {rid: load_surrogate_model(rid) for rid in run_ids}
    client = mlflow.MlflowClient()
    run_labels = {
        rid: client.get_run(rid).data.tags.get("mlflow.runName", rid[:6])
        for rid in run_ids
    }
    data = _sweep_amplitude_metrics(
        surrogates=surrogates,
        current_configs=current_configs,
        model_name=model_name,
        dt=dt,
        target_comp_name=comp_name,
        metric_key=metric_key,
    )
    return data, run_labels


def view_sweep(
    sweep_ui: mo.ui.dictionary,
    base_button: mo.ui.dictionary,
    param_button: mo.ui.dictionary,
    eval_ui: mo.ui.dictionary,
) -> mo.Html:
    """アダプター: marimo UI → run_sweep (計算) → _plot_sweep (描画) → mo.Html。"""
    ds = cast(dict[str, Any], base_button["base_dataset"].value)
    run_ids = cast(pd.DataFrame, base_button["run_selector"].value)[
        "run_id"
    ].tolist()
    comp_name = str(eval_ui["eval_comp"].value)
    metric_key = str(sweep_ui["metric"].value)
    data, run_labels = run_sweep(
        run_ids=run_ids,
        model_name=str(ds["model_name"]),
        dt=float(ds["dt"]),
        duration=float(ds["duration"]),
        silence_duration=float(ds["silence_duration"]),
        comp_name=comp_name,
        current_type=str(base_button["current_type"].value),
        base_current_params=cast(dict, param_button["current_params"].value),
        sweep_param=str(sweep_ui["sweep_param"].value),
        amp_start=float(cast(Any, sweep_ui["amp_start"]).value),
        amp_stop=float(cast(Any, sweep_ui["amp_stop"]).value),
        amp_steps=int(cast(Any, sweep_ui["amp_steps"]).value),
        metric_key=metric_key,
    )
    fig = _plot_sweep(data, run_ids, metric_key, comp_name, run_labels)
    return mo.vstack([mo.mpl.interactive(fig), mo.ui.table(data)])
