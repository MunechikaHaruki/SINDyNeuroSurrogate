import inspect
from collections import Counter

import marimo as mo
import pandas as pd
from analysis.access import current_of, target_of
from analysis.save.panel import SaveEntry, entry

from neurosurrogate.currents import CURRENT_MAP
from neurosurrogate.metrics.eval_sweep import CurrentSweepConfig, evaluate_sweep
from neurosurrogate.metrics.wave import DF_ROW_METRICS, SCALAR_METRICS
from neurosurrogate.surrogate.ansatz import NeuroSurrogateBase
from neurosurrogate.view.utils import sweep_fig, sweep_trace_grid_fig

# ---------------------------------------------------------------------------
# Sweep UI
# ---------------------------------------------------------------------------


def _sweep_param_of(current_type: str) -> str | None:
    """掃引軸に使う単一 numeric パラメータ名を返す。
    掃引軸が一意に定まらない (numeric param が 0 個 or 複数) 場合は
    None = 掃引 UI 非対応。"""
    numeric_params = [
        name
        for name, p in inspect.signature(CURRENT_MAP[current_type]).parameters.items()
        if p.annotation in (int, float)
    ]
    return numeric_params[0] if len(numeric_params) == 1 else None


def _is_sweepable(current_type: str) -> bool:
    return _sweep_param_of(current_type) is not None


SweepDefaults = dict[str, tuple[float, float, int]]
_SWEEP_FALLBACK = (-5.0, 20.0, 10)


def make_sweep_ui(
    current_type: str,
    defaults: SweepDefaults,
    run_selector: mo.ui.table,
    preset: dict | None = None,
) -> mo.ui.dictionary | None:
    if not _is_sweepable(current_type):
        return None
    start, stop, steps = defaults.get(current_type, _SWEEP_FALLBACK)
    p = preset or {}
    return mo.ui.dictionary(
        {
            "run_selector": run_selector,
            "amp_start": mo.ui.number(
                value=p.get("amp_start", start), step=1.0, label="amp_start"
            ),
            "amp_stop": mo.ui.number(
                value=p.get("amp_stop", stop), step=1.0, label="amp_stop"
            ),
            "amp_steps": mo.ui.number(
                value=p.get("amp_steps", steps), step=1, label="steps"
            ),
        }
    )


def make_draw_ui(
    base_ui: mo.ui.dictionary, preset: dict | None = None
) -> mo.ui.dictionary | None:
    current_type = current_of(base_ui)
    if not _is_sweepable(current_type):
        return None
    p = preset or {}
    ylim = p.get("ylim", {})
    metric = p.get("metric", "spike_count")
    options = DF_ROW_METRICS + SCALAR_METRICS
    return mo.ui.dictionary(
        {
            "metric": mo.ui.dropdown(
                options=options,
                value=metric if metric in options else "spike_count",
                label="metric",
            ),
            "ylim": mo.ui.dictionary(
                {
                    "auto": mo.ui.checkbox(value=ylim.get("auto", True), label="auto"),
                    "min": mo.ui.number(
                        value=ylim.get("min", 0.0), step=1.0, label="ymin"
                    ),
                    "max": mo.ui.number(
                        value=ylim.get("max", 1.0), step=1.0, label="ymax"
                    ),
                }
            ),
        }
    )


# ---------------------------------------------------------------------------
# Calc
# ---------------------------------------------------------------------------


def calc_sweep(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
    loaded: list[NeuroSurrogateBase],
) -> dict:
    """UI 値 + ロード済 surrogate を evaluate_sweep へ委譲。raw sim データを返す。
    surrogate は loaded (load_selected 由来) を単一源とし再取得しない。
    掃引結果の識別キーは meta.label。"""
    sweep_ui = setting_ui["sweep"]
    current_type = current_of(base_ui)
    cfg = CurrentSweepConfig(
        current_type=current_type,
        sweep_param=_sweep_param_of(current_type),
        amp_start=sweep_ui["amp_start"].value,
        amp_stop=sweep_ui["amp_stop"].value,
        amp_steps=sweep_ui["amp_steps"].value,
    )
    # label は掃引結果の識別キー。同 label 複数選択は dict 上書きで silent に
    # 1 run へ潰れ、summary 表と掃引図が食い違う → fail first で弾く。
    dup = [lbl for lbl, n in Counter(s.meta.label for s in loaded).items() if n > 1]
    if dup:
        raise ValueError(f"sweep 対象の meta.label 重複: {dup}。異なる config を選択。")
    sweep_eval = evaluate_sweep(
        {s.meta.label: s for s in loaded},
        model_name=target_of(base_ui),
        dt=float(base_ui["dt"].value),
        cfg=cfg,
    )
    return {"sweep_eval": sweep_eval, "cfg": cfg}


# ---------------------------------------------------------------------------
# View Result (save entry 列。表示は panel.render が担う)
# ---------------------------------------------------------------------------


def _eval_df(loaded: list[NeuroSurrogateBase]) -> pd.DataFrame:
    rows = [{"label": s.meta.label, **s.metrics()} for s in loaded]
    return pd.DataFrame(rows).set_index("label")


def view(
    loaded: list[NeuroSurrogateBase],
    res: dict | None,
    draw_ui: mo.ui.dictionary,
) -> list[SaveEntry]:
    """評価サマリ表 (選択 run) → sweep 波形格子 + メトリクス図 (res ゲート)。"""
    if not loaded:
        return []
    entries = [entry("eval_summary", _eval_df(loaded))]
    if res is None or "sweep" not in draw_ui:
        return entries
    labels = [s.meta.label for s in loaded]

    eval_comp = str(draw_ui["eval_comp"].value)
    ylim_ui = draw_ui["sweep"]["ylim"]
    ylim = (
        None
        if ylim_ui["auto"].value
        else (float(ylim_ui["min"].value), float(ylim_ui["max"].value))
    )
    metric_key = draw_ui["sweep"]["metric"].value
    data = res["sweep_eval"].metrics_df(eval_comp, metric_key)
    entries += [
        entry(
            "sweep_traces",
            sweep_trace_grid_fig(res["sweep_eval"], eval_comp, labels),
        ),
        entry(
            "sweep",
            sweep_fig(data, res["cfg"], eval_comp, metric_key, labels, ylim=ylim),
        ),
    ]
    return entries
