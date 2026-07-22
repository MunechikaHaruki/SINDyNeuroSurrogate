import inspect
import typing
from typing import Literal

import marimo as mo
from analysis.access import (
    current_of,
    dt_of,
    eval_comp_of,
    sim_current_params_of,
    sim_run_selection_of,
    target_of,
)
from analysis.save.panel import SaveEntry, entry
from matplotlib.figure import Figure
from mlflow_io import load_surrogate_model

from neurosurrogate.core.network import DatasetConfig, NeuronGraph
from neurosurrogate.currents import CURRENT_MAP
from neurosurrogate.metrics.eval import EvalResult, evaluate
from neurosurrogate.models import MCMODELS
from neurosurrogate.surrogate.bundle import SurrogateBundle
from neurosurrogate.surrogate.replace import replaced_names
from neurosurrogate.view.model import closure_figs, view_neuron_graph
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
        # step 指定なし → 任意精度で入力可 (0.1 刻み制限を外す。例: 1e-4/area=3.012…)
        return mo.ui.number(value=float(default), label=name)
    elif annotation is bool:
        return mo.ui.checkbox(value=bool(default), label=name)
    elif annotation is list:
        vals = default if isinstance(default, list) and default else [0.0]
        return mo.ui.array(
            [mo.ui.number(value=float(v), step=0.1) for v in vals], label=name
        )
    else:
        raise NotImplementedError(f"{name}: {annotation} は未対応の型です")


def make_draw_ui(spike: dict | None = None) -> mo.ui.dictionary:
    s = spike or {}
    return mo.ui.dictionary(
        {
            "spike": mo.ui.dictionary(
                {
                    "orig": mo.ui.number(
                        value=int(s.get("orig", 0)), step=1, label="spike orig #"
                    ),
                    "surr": mo.ui.number(
                        value=int(s.get("surr", 0)), step=1, label="spike surr #"
                    ),
                }
            ),
        }
    )


def make_sim_ui(
    current_type: str,
    run_selector: mo.ui.table,
    current_params: dict | None = None,
) -> mo.ui.dictionary:
    cp = current_params or {}
    current_params_ui = mo.ui.dictionary(
        {
            name: _make_ui_element(
                name,
                param.annotation,
                cp.get(
                    name,
                    0 if param.default is inspect.Parameter.empty else param.default,
                ),
            )
            for name, param in inspect.signature(
                CURRENT_MAP[current_type]
            ).parameters.items()
        }
    )
    return mo.ui.dictionary(
        {"run_selector": run_selector, "current_params": current_params_ui}
    )


# ---------------------------------------------------------------------------
# Calc Eval
# ---------------------------------------------------------------------------


def calc_eval(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
) -> EvalResult:
    run_ids = sim_run_selection_of(setting_ui)["run_id"].tolist()
    if len(run_ids) != 1:
        raise ValueError(
            f"single モードでは Run を 1 件だけ選択。現在: {len(run_ids)} 件"
        )
    dataset_cfg = DatasetConfig.build_dataset(
        model_name=target_of(base_ui),
        dt=dt_of(base_ui),
        current_type=current_of(base_ui),
        current_params=sim_current_params_of(setting_ui),
    )
    return evaluate(load_surrogate_model(str(run_ids[0])), dataset_cfg)


# ---------------------------------------------------------------------------
# Model View (eval 前・静的: surrogate + net のみ依存)
# ---------------------------------------------------------------------------


def model_figs(
    net: NeuronGraph, surrogate: SurrogateBundle
) -> list[tuple[str, Figure]]:
    """single mode の静的モデル図。(save 名, fig) 列。"""
    return [
        ("neurograph", view_neuron_graph(net, replaced_names(surrogate.meta, net))),
        *closure_figs(surrogate.closure),
    ]


# ---------------------------------------------------------------------------
# View Result (save entry 列。表示は panel.render が担う)
# ---------------------------------------------------------------------------


def view(
    surrogate: SurrogateBundle | None,
    base_ui: mo.ui.dictionary,
    res: EvalResult | None,
    draw_ui: mo.ui.dictionary,
) -> list[SaveEntry]:
    """静的モデル図 (選択 run) → 波形図 + メトリクス df (res ゲート)。"""
    if surrogate is None:
        return []
    net = MCMODELS[target_of(base_ui)]
    entries = [entry(name, fig) for name, fig in model_figs(net, surrogate)]
    if res is None:
        return entries

    target_comp_id = res.dataset.net.name_to_idx(eval_comp_of(draw_ui))
    spike_orig = int(draw_ui["single"]["spike"]["orig"].value)
    spike_surr = int(draw_ui["single"]["spike"]["surr"].value)
    rep = res.wave_report(target_comp_id, spike_orig, spike_surr)

    entries += [entry(name, fig) for name, fig in draw_all(res, target_comp_id)]
    entries += [
        entry("metrics", rep.df_metrics),
        entry("metrics_scalar", rep.df_scalar),
    ]
    return entries
