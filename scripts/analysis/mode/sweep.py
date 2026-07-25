import marimo as mo
import pandas as pd
from analysis.access import (
    comp_type_of,
    current_type_of,
    dt_of,
    eval_comp_of,
    sweep_config_inputs,
    valid_or,
)
from analysis.save.panel import SaveEntry, entry
from analysis.targets import TARGET_MODEL

from neurosurrogate.metrics.eval_sweep import (
    CurrentSweepConfig,
    evaluate_sweep,
    sweep_labels,
    sweepable_params,
)
from neurosurrogate.metrics.wave import DF_ROW_METRICS, SCALAR_METRICS
from neurosurrogate.surrogate.bundle import SurrogateBundle
from neurosurrogate.view.preview import sweep_fig, sweep_trace_grid_fig

# ---------------------------------------------------------------------------
# Sweep UI (amp 範囲は base.json 側 = widget 無し。表示調整の draw_ui だけ残す)
# ---------------------------------------------------------------------------


def is_sweepable(cfg: dict) -> bool:
    """cfg の電流が amp 掃引できるか (sweep 実行ボタン/描画の出し分けに使う)。"""
    return len(sweepable_params(current_type_of(cfg))) > 0


def draw_fields(cfg: dict, p: dict) -> dict | None:
    """sweep 表示設定 (flat な key の dict。draw_ui へ merge される)。sweep 非対応
    電流なら None。metric カタログ + sweepable 判定という sweep ドメイン知識をここに
    集約する (ui は返った dict をそのまま広げるだけ)。"""
    if not is_sweepable(cfg):
        return None
    options = DF_ROW_METRICS + SCALAR_METRICS
    return {
        "sweep_metric": mo.ui.dropdown(
            options=options,
            value=valid_or(p.get("sweep_metric"), options, "spike_count"),
            label="sweep metric",
        ),
        "sweep_yauto": mo.ui.checkbox(value=p.get("sweep_yauto", True), label="y auto"),
        "sweep_ymin": mo.ui.number(
            value=p.get("sweep_ymin", 0.0), step=1.0, label="ymin"
        ),
        "sweep_ymax": mo.ui.number(
            value=p.get("sweep_ymax", 1.0), step=1.0, label="ymax"
        ),
    }


# ---------------------------------------------------------------------------
# Calc
# ---------------------------------------------------------------------------


def calc_sweep(
    cfg: dict,
    loaded: list[SurrogateBundle],
) -> dict:
    """cfg (base.json⊕meta.json) の掃引設定 + ロード済 surrogate を evaluate_sweep へ
    委譲。raw sim データを返す。surrogate は loaded (sweep 兄弟) を単一源とし再取得
    しない。target/comp_type は loaded の meta から自動決定。掃引結果の識別キーは
    label。"""
    sweep_cfg = CurrentSweepConfig(**sweep_config_inputs(cfg))
    # sweep は全 target は回さず代表 target (TARGET_MODEL[comp_type][0]) 1 つ。
    # 兄弟 run × amp を既に掛けており target 軸を足すと図が過剰になるため。
    # comp_type は兄弟 run 共通なので先頭 loaded の meta から取る。
    sweep_eval = evaluate_sweep(
        dict(zip(sweep_labels(loaded), loaded, strict=True)),
        model_name=TARGET_MODEL[comp_type_of(loaded[0].meta)][0],
        dt=dt_of(cfg),
        cfg=sweep_cfg,
    )
    return {"sweep_eval": sweep_eval, "cfg": sweep_cfg}


# ---------------------------------------------------------------------------
# View Result (save entry 列。表示は panel.render が担う)
# ---------------------------------------------------------------------------


def _eval_df(loaded: list[SurrogateBundle]) -> pd.DataFrame:
    rows = [
        {"label": lbl, **s.metrics()}
        for lbl, s in zip(sweep_labels(loaded), loaded, strict=True)
    ]
    return pd.DataFrame(rows).set_index("label")


def view(
    loaded: list[SurrogateBundle],
    res: dict | None,
    draw: dict,
) -> list[SaveEntry]:
    """評価サマリ表 (選択 run) → sweep 波形格子 + メトリクス図 (res ゲート)。"""
    if not loaded:
        return []
    entries = [entry("eval_summary", _eval_df(loaded))]
    if res is None or "sweep_metric" not in draw:
        return entries
    labels = sweep_labels(loaded)  # calc_sweep が dict キーに使ったものと同一規則

    eval_comp = eval_comp_of(draw)
    ylim = (
        None
        if draw["sweep_yauto"]
        else (float(draw["sweep_ymin"]), float(draw["sweep_ymax"]))
    )
    metric_key = draw["sweep_metric"]
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
