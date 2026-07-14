import inspect
import typing
from typing import Literal, cast

import marimo as mo
import pandas as pd
from matplotlib.figure import Figure
from mlflow_io import load_surrogate_model, sole_target_model

from neurosurrogate.core.network import DatasetConfig
from neurosurrogate.core.simulator import unified_simulator
from neurosurrogate.currents import CURRENT_MAP
from neurosurrogate.metrics.wave import (
    DynamicMetrics,
    n_spikes,
    spike_features_df,
    spike_shape_corr,
    waveform_summary,
    waveform_summary_df,
)
from neurosurrogate.models import MCMODELS
from neurosurrogate.surrogate import preprocessed_latent
from neurosurrogate.view.specs import draw_all

# ---------------------------------------------------------------------------
# Sim UI
# ---------------------------------------------------------------------------


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


def make_draw_ui() -> mo.ui.dictionary:
    return mo.ui.dictionary(
        {
            "spike": mo.ui.dictionary(
                {
                    "orig": mo.ui.number(value=0, step=1, label="spike orig #"),
                    "surr": mo.ui.number(value=0, step=1, label="spike surr #"),
                }
            ),
        }
    )


def make_sim_ui(current_type: str) -> mo.ui.dictionary:
    current_params_ui = mo.ui.dictionary(
        {
            name: _make_ui_element(
                name,
                param.annotation,
                0 if param.default is inspect.Parameter.empty else param.default,
            )
            for name, param in inspect.signature(
                CURRENT_MAP[current_type]
            ).parameters.items()
        }
    )
    return mo.ui.dictionary({"current_params": current_params_ui})


# ---------------------------------------------------------------------------
# Calc Eval
# ---------------------------------------------------------------------------


def _parse_eval_button(
    base_ui: mo.ui.dictionary,
    sim_ui: mo.ui.dictionary,
) -> tuple[DatasetConfig, str]:
    selected = cast(pd.DataFrame, base_ui["run_selector"].value)
    run_ids = selected["run_id"].tolist()
    if len(run_ids) != 1:
        raise ValueError(
            f"single モードでは Run を 1 件だけ選択。現在: {len(run_ids)} 件"
        )
    current_type = str(base_ui["sim_current_type"].value)
    current_params = sim_ui["current_params"].value or {}
    dataset_cfg = DatasetConfig.build_dataset(
        model_name=sole_target_model(selected),
        dt=float(base_ui["dt"].value),
        current={"type": current_type, "params": current_params},
    )
    return dataset_cfg, str(run_ids[0])


def calc_eval(
    base_ui: mo.ui.dictionary,
    sim_ui: mo.ui.dictionary,
) -> dict:
    dataset_cfg, run_id = _parse_eval_button(base_ui, sim_ui)

    surrogate_model = load_surrogate_model(run_id)
    original_ds = unified_simulator(dataset_cfg)
    surr_ds = unified_simulator(surrogate_model.apply(dataset_cfg))

    return {
        "original_ds": original_ds,
        "surr_ds": surr_ds,
        "dt": dataset_cfg.dt,
        "get_preprocessed": lambda comp_id: preprocessed_latent(
            surrogate_model, dataset_cfg, original_ds, comp_id
        ),
        "name_to_idx": MCMODELS[dataset_cfg.model_name].name_to_idx,
        "make_dm": lambda comp_id: DynamicMetrics(
            original_ds, surr_ds, comp_id, dataset_cfg.dt
        ),
    }


# ---------------------------------------------------------------------------
# View Result
# ---------------------------------------------------------------------------


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
    eval_comp_name: str,
) -> tuple[mo.Html, list[tuple[str, Figure]], dict[str, pd.DataFrame]]:
    target_comp_id = result["name_to_idx"](eval_comp_name)
    dm = result["make_dm"](target_comp_id)

    n_orig_count, n_surr_count = n_spikes(dm)
    spike_orig = int(draw_ui["spike"]["orig"].value)
    spike_surr = int(draw_ui["spike"]["surr"].value)
    has_valid_spikes = 0 <= spike_orig < n_orig_count and 0 <= spike_surr < n_surr_count

    wf_summary = waveform_summary(dm)
    df_waveform = waveform_summary_df(dm)

    figs = draw_all(
        result["original_ds"],
        result["surr_ds"],
        target_comp_id,
        lambda: result["get_preprocessed"](target_comp_id),
    )

    html_parts: list = [
        mo.md("#### 波形誤差スカラー"),
        _stat_cards(wf_summary),
    ]
    df_metrics = df_waveform
    scalar_data: dict = dict(wf_summary)

    if has_valid_spikes:
        spike_corr = spike_shape_corr(dm)
        df_spike = spike_features_df(dm, spike_orig=spike_orig, spike_surr=spike_surr)
        df_spike.index.name = "metric"
        df_metrics = pd.concat([df_waveform, df_spike])
        scalar_data.update(spike_corr)
        html_parts += [
            mo.md(
                f"#### 動的指標（orig / surr / orig-surr）"
                f" — spike orig: {spike_orig} / surr: {spike_surr}"
            ),
            df_metrics,
            mo.md("#### スパイク波形相関（spike_shape_corr）"),
            _stat_cards(spike_corr),
        ]
    else:
        html_parts.append(
            mo.md(
                f"（スパイク指標なし: orig {spike_orig}/{n_orig_count}"
                f" surr {spike_surr}/{n_surr_count}）"
            )
        )

    for name, fig in figs:
        html_parts.append(mo.md(f"##### {name}"))
        html_parts.append(mo.mpl.interactive(fig))

    df_scalar = pd.DataFrame(
        scalar_data.items(), columns=["metric", "value"]
    ).set_index("metric")

    return (
        mo.vstack(html_parts),
        figs,
        {
            "metrics": df_metrics,
            "metrics(scalar)": df_scalar,
        },
    )
