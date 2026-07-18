import inspect

import marimo as mo
from matplotlib.figure import Figure
from mlflow_io import LoadedRun

from neurosurrogate.currents import CURRENT_MAP
from neurosurrogate.metrics.eval_sweep import CurrentSweepConfig, evaluate_sweep
from neurosurrogate.metrics.wave import DF_ROW_METRICS, SCALAR_METRICS
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

# runName は `<preset> key=value ...` 形式 (main._make_run_name)。長く図で重なるため
# value=... だけ残し n_components は n<val> に畳んで短縮する。preset 名 (hh 等) は
# 同一 sweep で混在比較しないため落とす。type= は2つ (surrogate 種 / preprocessor 種)
# あるが値自体で区別できるため key は落として良い。
_ABBR_VALUE = {"hybrid": "hybrid", "sindy": "sindy"}


def _fetch_abbr_name(run_name: str) -> str:
    """runName をパースし図表示用の簡約名にする (runName 本体は不変)。例:
    `hh type=hybrid n_components=2 type=ae` → `hybrid/n2/ae`。"""
    out: list[str] = []
    for tok in run_name.split():
        if "=" not in tok:  # preset 名等はスキップ
            continue
        key, val = tok.split("=", 1)
        out.append(f"n{val}" if key == "n_components" else _ABBR_VALUE.get(val, val))
    return "/".join(out) if out else run_name


def make_sweep_ui(
    current_type: str, defaults: SweepDefaults, run_selector: mo.ui.table
) -> mo.ui.dictionary | None:
    if not _is_sweepable(current_type):
        return None
    start, stop, steps = defaults.get(current_type, _SWEEP_FALLBACK)
    return mo.ui.dictionary(
        {
            "run_selector": run_selector,
            "amp_start": mo.ui.number(value=start, step=1.0, label="amp_start"),
            "amp_stop": mo.ui.number(value=stop, step=1.0, label="amp_stop"),
            "amp_steps": mo.ui.number(value=steps, step=1, label="steps"),
        }
    )


def make_draw_ui(base_ui: mo.ui.dictionary) -> mo.ui.dictionary | None:
    current_type = str(base_ui["sim_current_type"].value)
    if not _is_sweepable(current_type):
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


def calc_sweep(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
    loaded: list[LoadedRun],
) -> dict:
    """UI 値 + ロード済 run を evaluate_sweep へ委譲。raw sim データを返す。
    surrogate/runName は loaded (load_selected 由来) を単一源とし再取得しない。"""
    sweep_ui = setting_ui["sweep"]
    current_type = base_ui["sim_current_type"].value
    cfg = CurrentSweepConfig(
        current_type=current_type,
        sweep_param=_sweep_param_of(current_type),
        amp_start=sweep_ui["amp_start"].value,
        amp_stop=sweep_ui["amp_stop"].value,
        amp_steps=sweep_ui["amp_steps"].value,
    )
    sweep_eval = evaluate_sweep(
        {r.run_id: r.surrogate for r in loaded},
        model_name=str(base_ui["model_pair"].value[1]),
        dt=float(base_ui["dt"].value),
        cfg=cfg,
    )
    return {
        "sweep_eval": sweep_eval,
        "run_labels": {r.run_id: _fetch_abbr_name(r.run_name) for r in loaded},
        "cfg": cfg,
    }


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------


def plot_sweep(
    sweep_raw: dict,
    eval_comp_name: str,
    metric_key: str,
    ylim: tuple[float, float] | None = None,
) -> tuple[mo.Html, Figure]:
    """描画層: SweepEval からメトリクス計算 → 描画。シミュ再走なし。"""
    data = sweep_raw["sweep_eval"].metrics_df(eval_comp_name, metric_key)
    fig = sweep_fig(
        data,
        sweep_raw["cfg"],
        eval_comp_name,
        metric_key,
        sweep_raw["run_labels"],
        ylim=ylim,
    )
    return mo.vstack([mo.mpl.interactive(fig), mo.ui.table(data)]), fig


def plot_sweep_traces(
    sweep_raw: dict,
    eval_comp_name: str,
) -> tuple[mo.Html, Figure]:
    """列=掃引 amp の波形格子 (行1=I_ext / 行2以降=各 run vs orig)。再走なし。"""
    fig = sweep_trace_grid_fig(
        sweep_raw["sweep_eval"],
        eval_comp_name,
        sweep_raw["run_labels"],
    )
    return mo.mpl.interactive(fig), fig
