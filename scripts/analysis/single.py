import inspect
import typing
from typing import Literal, cast

import marimo as mo
import pandas as pd
from matplotlib.figure import Figure
from mlflow_io import load_surrogate_model, sole_target_model

from neurosurrogate.core.network import DatasetConfig
from neurosurrogate.currents import CURRENT_MAP
from neurosurrogate.metrics.eval import EvalResult, evaluate
from neurosurrogate.metrics.wave import WaveReport
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


def calc_eval(
    base_ui: mo.ui.dictionary,
    sim_ui: mo.ui.dictionary,
) -> EvalResult:
    selected = cast(pd.DataFrame, base_ui["run_selector"].value)
    run_ids = selected["run_id"].tolist()
    if len(run_ids) != 1:
        raise ValueError(
            f"single モードでは Run を 1 件だけ選択。現在: {len(run_ids)} 件"
        )
    dataset_cfg = DatasetConfig.build_dataset(
        model_name=sole_target_model(selected),
        dt=float(base_ui["dt"].value),
        current={
            "type": str(base_ui["sim_current_type"].value),
            "params": sim_ui["current_params"].value or {},
        },
    )
    return evaluate(load_surrogate_model(str(run_ids[0])), dataset_cfg)


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


def _metrics_html(rep: WaveReport, spike_orig: int, spike_surr: int) -> mo.Html:
    """WaveReport をメトリクス表示 UI に組み立てる。"""
    parts: list = [
        mo.md("#### 波形誤差スカラー"),
        _stat_cards(rep.waveform_scalar),
    ]
    if rep.has_spikes:
        parts += [
            mo.md(
                f"#### 動的指標（orig / surr / orig-surr）"
                f" — spike orig: {spike_orig} / surr: {spike_surr}"
            ),
            rep.df_metrics,
            mo.md("#### スパイク波形相関（spike_shape_corr）"),
            _stat_cards(rep.spike_shape_corr),
        ]
    else:
        parts.append(
            mo.md(
                f"（スパイク指標なし: orig {spike_orig}/{rep.n_orig}"
                f" surr {spike_surr}/{rep.n_surr}）"
            )
        )
    return mo.vstack(parts)


def view_result(
    draw_ui: mo.ui.dictionary,
    result: EvalResult,
    eval_comp_name: str,
) -> tuple[mo.Html, list[tuple[str, Figure]], dict[str, pd.DataFrame]]:
    target_comp_id = result.name_to_idx(eval_comp_name)
    spike_orig = int(draw_ui["spike"]["orig"].value)
    spike_surr = int(draw_ui["spike"]["surr"].value)
    rep = result.wave_report(target_comp_id, spike_orig, spike_surr)

    figs = draw_all(
        result.original_ds,
        result.surr_ds,
        target_comp_id,
        lambda: result.preprocessed_latent(target_comp_id),
    )
    fig_html = [
        part
        for name, fig in figs
        for part in (mo.md(f"##### {name}"), mo.mpl.interactive(fig))
    ]

    return (
        mo.vstack([_metrics_html(rep, spike_orig, spike_surr), *fig_html]),
        figs,
        {
            "metrics": rep.df_metrics,
            "metrics(scalar)": rep.df_scalar,
        },
    )
