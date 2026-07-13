import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import marimo as mo
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mlflow_io import load_surrogate_model

from neurosurrogate.core.network import DatasetConfig
from neurosurrogate.core.simulator import unified_simulator
from neurosurrogate.currents import CURRENT_MAP
from neurosurrogate.metrics.wave import (
    DF_ROW_METRICS,
    SCALAR_METRICS,
    DynamicMetrics,
    extract_metric,
)
from neurosurrogate.models import MCMODELS

# ---------------------------------------------------------------------------
# Sweep UI
# ---------------------------------------------------------------------------


def _sweep_param_of(current_type: str) -> str | None:
    keys = [
        n
        for n, p in inspect.signature(CURRENT_MAP[current_type]).parameters.items()
        if p.annotation in (int, float)
    ]
    return keys[0] if len(keys) == 1 else None


SweepDefaults = dict[str, tuple[float, float, int]]
_SWEEP_FALLBACK = (-5.0, 20.0, 10)


def make_sweep_ui(
    current_type: str, defaults: SweepDefaults
) -> mo.ui.dictionary | None:
    if _sweep_param_of(current_type) is None:
        return None
    start, stop, steps = defaults.get(current_type, _SWEEP_FALLBACK)
    return mo.ui.dictionary(
        {
            "amp_start": mo.ui.number(value=start, step=1.0, label="amp_start"),
            "amp_stop": mo.ui.number(value=stop, step=1.0, label="amp_stop"),
            "amp_steps": mo.ui.number(value=steps, step=1, label="steps"),
        }
    )


def make_draw_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary | None:
    current_type = str(base_ui["sim_current_type"].value)
    if _sweep_param_of(current_type) is None:
        return None
    return mo.ui.dictionary(
        {
            "metric": mo.ui.dropdown(
                options=DF_ROW_METRICS + SCALAR_METRICS,
                value="spike_count",
                label="metric",
            ),
            "ylim": mo.ui.dictionary(
                {
                    "auto": mo.ui.checkbox(value=True, label="auto"),
                    "min": mo.ui.number(value=0.0, step=1.0, label="ymin"),
                    "max": mo.ui.number(value=1.0, step=1.0, label="ymax"),
                }
            ),
        }
    )


# ---------------------------------------------------------------------------
# Calc
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    amp_start: float
    amp_stop: float
    amp_steps: int

    @property
    def amp_values(self) -> np.ndarray:
        return np.linspace(self.amp_start, self.amp_stop, self.amp_steps)


def _iter_amp_datasets(
    surrogates: dict,
    current_configs: dict[float, dict],
    model_name: str,
    dt: float,
) -> Iterator[tuple[float, Any, dict[str, Any]]]:
    """各 amp で (amp, orig_ds, {rid: surr_ds}) を yield。eval_comp 不要。"""
    net = MCMODELS[model_name]
    for amp, current in current_configs.items():
        dset_cfg = DatasetConfig(model_name=model_name, dt=dt, current=current, net=net)
        orig_ds = unified_simulator(dset_cfg)
        surr_datasets: dict[str, Any] = {
            rid: unified_simulator(surrogate.apply(dset_cfg))
            for rid, surrogate in surrogates.items()
        }
        yield amp, orig_ds, surr_datasets


def _run_sweep(
    *,
    run_ids: list[str],
    model_name: str,
    dt: float,
    current_type: str,
    sweep_param: str,
    cfg: SweepConfig,
) -> tuple[list[tuple[float, Any, dict[str, Any]]], dict[str, str]]:
    """純粋計算層: raw amp_datasets と run_label dict を返す。plot しない。"""
    current_configs: dict[float, dict] = {
        amp: {"type": current_type, "params": {sweep_param: amp}}
        for amp in cfg.amp_values
    }
    surrogates = {rid: load_surrogate_model(rid) for rid in run_ids}
    run_labels = {
        rid: mlflow.MlflowClient().get_run(rid).data.tags.get("mlflow.runName", rid[:6])
        for rid in run_ids
    }
    amp_datasets = list(
        _iter_amp_datasets(
            surrogates=surrogates,
            current_configs=current_configs,
            model_name=model_name,
            dt=dt,
        )
    )
    return amp_datasets, run_labels


def calc_sweep(
    base_button: mo.ui.dictionary,
    sweep_ui: mo.ui.dictionary,
) -> dict:
    """純粋実行層: draw_ui 不要。raw sim データを返す。"""
    model_name = base_button["model_name"].value
    run_ids = base_button["run_selector"].value["run_id"].tolist()
    current_type = base_button["sim_current_type"].value
    sweep_param = _sweep_param_of(current_type)
    sweep_cfg = SweepConfig(
        amp_start=sweep_ui["amp_start"].value,
        amp_stop=sweep_ui["amp_stop"].value,
        amp_steps=sweep_ui["amp_steps"].value,
    )
    dt = float(base_button["dt"].value)
    amp_datasets, run_labels = _run_sweep(
        run_ids=run_ids,
        model_name=model_name,
        dt=dt,
        current_type=current_type,
        sweep_param=sweep_param,
        cfg=sweep_cfg,
    )
    return {
        "amp_datasets": amp_datasets,
        "run_labels": run_labels,
        "run_ids": run_ids,
        "sweep_param": sweep_param,
        "model_name": model_name,
        "dt": dt,
    }


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------


def _compute_metrics_df(
    amp_datasets: list[tuple[float, Any, dict[str, Any]]],
    eval_comp_name: str,
    model_name: str,
    dt: float,
    metric_key: str,
) -> pd.DataFrame:
    """plot時呼び出し: eval_comp_name + metric_key でメトリクス DataFrame 構築。"""
    net = MCMODELS[model_name]
    eval_comp_id = net.name_to_idx(eval_comp_name)
    rows: list[dict] = []
    for amp, orig_ds, surr_datasets in amp_datasets:
        dms = {
            rid: DynamicMetrics(orig_ds, surr_ds, eval_comp_id, dt)
            for rid, surr_ds in surr_datasets.items()
        }
        extracted = {rid: extract_metric(dm, metric_key) for rid, dm in dms.items()}
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
    sweep_param: str,
    run_labels: dict[str, str] | None = None,
    ylim: tuple[float, float] | None = None,
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

    ax.set_xlabel(sweep_param)
    ax.set_ylabel(metric_key)
    ax.set_title(f"{sweep_param} sweep — {metric_key} ({comp_name})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sweep(
    sweep_raw: dict,
    eval_comp_name: str,
    metric_key: str,
    ylim: tuple[float, float] | None = None,
) -> tuple[mo.Html, Figure]:
    """描画層: eval_comp_name + metric_key でメトリクス計算 → 描画。シミュ再走なし。"""
    data = _compute_metrics_df(
        sweep_raw["amp_datasets"],
        eval_comp_name=eval_comp_name,
        model_name=sweep_raw["model_name"],
        dt=sweep_raw["dt"],
        metric_key=metric_key,
    )
    fig = _plot_sweep(
        data,
        sweep_raw["run_ids"],
        metric_key,
        eval_comp_name,
        sweep_param=sweep_raw["sweep_param"],
        run_labels=sweep_raw["run_labels"],
        ylim=ylim,
    )
    return mo.vstack([mo.mpl.interactive(fig), mo.ui.table(data)]), fig
