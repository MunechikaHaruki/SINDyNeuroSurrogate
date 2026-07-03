import inspect
from collections.abc import Iterator
from dataclasses import dataclass
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
from neurosurrogate.model.model_dataset import CurrentConfig, DatasetConfig
from neurosurrogate.model.registry_neuron import MCMODELS
from neurosurrogate.profiler.profiler_wave import (
    DF_ROW_METRICS,
    SCALAR_METRICS,
    DynamicMetrics,
    extract_metric,
)


@dataclass(frozen=True)
class SweepConfig:
    amp_start: float
    amp_stop: float
    amp_steps: int
    metric_key: str

    @property
    def amp_values(self) -> np.ndarray:
        return np.linspace(self.amp_start, self.amp_stop, self.amp_steps)


# ---------------------------------------------------------------------------
# Sweep UI
# ---------------------------------------------------------------------------


def _sweep_param_of(current_type: str) -> str | None:
    param_keys = [
        name
        for name, p in inspect.signature(FUNC_MAP[current_type]).parameters.items()
        if p.annotation in (int, float)
    ]
    return param_keys[0] if len(param_keys) == 1 else None


def make_sweep_ui(current_type: str) -> mo.ui.dictionary | None:
    if _sweep_param_of(current_type) is None:
        return None
    return mo.ui.dictionary(
        {
            "amp_start": mo.ui.number(value=-5.0, step=1.0, label="amp_start"),
            "amp_stop": mo.ui.number(value=20.0, step=1.0, label="amp_stop"),
            "amp_steps": mo.ui.number(value=10, step=1, label="steps"),
            "metric": mo.ui.dropdown(
                options=DF_ROW_METRICS + SCALAR_METRICS, value="spike_count"
            ),
        }
    )


def render_sweep(sweep_ui: mo.ui.dictionary) -> mo.Html:
    return mo.vstack([mo.md("### 振幅スイープ設定"), mo.md(f"{sweep_ui}")])


# ---------------------------------------------------------------------------
# Run and Sweep
# ---------------------------------------------------------------------------


def _iter_amp_dms(
    surrogates: dict,
    current_configs: dict[float, CurrentConfig],
    model_name: str,
    dt: float,
    target_comp_names: list[str],
    eval_comp_name: str,
) -> Iterator[tuple[float, dict[str, DynamicMetrics]]]:
    """各 amp で {rid: DynamicMetrics} を yield。orig_ds は amp ごとに1回計算。"""
    net = MCMODELS[model_name]
    eval_comp_id = net.name_to_idx(eval_comp_name)
    for amp, current in current_configs.items():
        dset_cfg = DatasetConfig(model_name=model_name, dt=dt, current=current, net=net)
        orig_ds = unified_simulator(dset_cfg)
        dms: dict[str, DynamicMetrics] = {}
        for rid, surrogate in surrogates.items():
            surr_ds = unified_simulator(
                dset_cfg.with_surrogates(
                    targets=set(target_comp_names),
                    make_surr=surrogate.make_surr_comp,
                ),
                surrogate_model=surrogate,
            )
            dms[rid] = DynamicMetrics(orig_ds, surr_ds, eval_comp_id, dt)
        yield amp, dms


def _sweep_amplitude_metrics(
    surrogates: dict,
    current_configs: dict[float, CurrentConfig],
    model_name: str,
    dt: float,
    target_comp_names: list[str],
    eval_comp_name: str,
    metric_key: str,
) -> pd.DataFrame:
    """amp × rid を走査し metric DataFrame を構築。

    orig 列の有無は extract_metric の戻り値 (None かどうか) で決まる。
    """
    rows: list[dict] = []
    for amp, dms in _iter_amp_dms(
        surrogates, current_configs, model_name, dt, target_comp_names, eval_comp_name
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
    sweep_param: str,
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

    ax.set_xlabel(sweep_param)
    ax.set_ylabel(metric_key)
    ax.set_title(f"{sweep_param} sweep — {metric_key} ({comp_name})")
    ax.legend()
    fig.tight_layout()
    return fig


def _run_sweep(
    *,
    run_ids: list[str],
    model_name: str,
    dt: float,
    target_comp_names: list[str],
    eval_comp_name: str,
    current_type: str,
    sweep_param: str,
    cfg: SweepConfig,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """純粋計算層: metric DataFrame と run_label dict を返す。plot しない。"""
    current_configs: dict[float, CurrentConfig] = {
        amp: CurrentConfig(
            pipeline=CurrentConfig.build_pipeline(current_type, {sweep_param: amp}),
        )
        for amp in cfg.amp_values
    }
    surrogates = {rid: load_surrogate_model(rid) for rid in run_ids}
    run_labels = {
        rid: mlflow.MlflowClient().get_run(rid).data.tags.get("mlflow.runName", rid[:6])
        for rid in run_ids
    }
    data = _sweep_amplitude_metrics(
        surrogates=surrogates,
        current_configs=current_configs,
        model_name=model_name,
        dt=dt,
        target_comp_names=target_comp_names,
        eval_comp_name=eval_comp_name,
        metric_key=cfg.metric_key,
    )
    return data, run_labels


def _ui_val(ui: mo.ui.dictionary, key: str) -> Any:
    return cast(Any, ui[key]).value


def calc_sweep(
    base_button: mo.ui.dictionary,
    sweep_ui: mo.ui.dictionary,
    surrogate_targets: list[str],
    draw_ui: mo.ui.dictionary,
) -> dict:
    """純粋実行層: marimo UI → _run_sweep → 結果dict。"""
    model_name = str(_ui_val(base_button, "model_name"))
    run_ids = cast(pd.DataFrame, base_button["run_selector"].value)["run_id"].tolist()
    eval_comp_name = str(_ui_val(draw_ui, "eval_comp"))
    current_type = str(_ui_val(base_button, "sim_current_type"))
    sweep_param = cast(str, _sweep_param_of(current_type))
    sweep_cfg = SweepConfig(
        amp_start=float(_ui_val(sweep_ui, "amp_start")),
        amp_stop=float(_ui_val(sweep_ui, "amp_stop")),
        amp_steps=int(_ui_val(sweep_ui, "amp_steps")),
        metric_key=str(_ui_val(sweep_ui, "metric")),
    )
    data, run_labels = _run_sweep(
        run_ids=run_ids,
        model_name=model_name,
        dt=float(base_button["dt"].value),
        target_comp_names=surrogate_targets,
        eval_comp_name=eval_comp_name,
        current_type=current_type,
        sweep_param=sweep_param,
        cfg=sweep_cfg,
    )
    return {
        "data": data,
        "run_labels": run_labels,
        "run_ids": run_ids,
        "metric_key": sweep_cfg.metric_key,
        "sweep_param": sweep_param,
        "eval_comp_name": eval_comp_name,
    }


def plot_sweep(sweep_result: dict) -> tuple[mo.Html, Figure | None]:
    """純粋描画層: 結果dict → (Html, Figure)。"""
    if not sweep_result:
        return mo.md(""), None
    fig = _plot_sweep(
        sweep_result["data"],
        sweep_result["run_ids"],
        sweep_result["metric_key"],
        sweep_result["eval_comp_name"],
        sweep_param=sweep_result["sweep_param"],
        run_labels=sweep_result["run_labels"],
    )
    return mo.vstack([mo.mpl.interactive(fig), mo.ui.table(sweep_result["data"])]), fig
