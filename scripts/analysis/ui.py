from __future__ import annotations

from typing import cast

import marimo as mo
import pandas as pd
from analysis.access import (
    ALL_PRESETS,
    comp_type_of,
    current_inputs,
    preset_of,
    valid_or,
)
from analysis.mode import single as analysis_single
from analysis.mode import sweep as analysis_sweep
from analysis.save.panel import SaveEntry
from analysis.style import STYLES
from analysis.targets import TARGET_MODEL
from mlflow_io import setup_mlflow

from neurosurrogate.metrics.eval import EvalResult
from neurosurrogate.neurons import MCMODELS
from neurosurrogate.surrogate.bundle import SurrogateBundle
from neurosurrogate.view.preview import current_preview_fig

setup_mlflow()


# ---------------------------------------------------------------------------
# Preset filter + Run 選択 (marimo に残す唯一の「入力」widget)
# ---------------------------------------------------------------------------


def make_preset_ui(runs_df: pd.DataFrame, cfg: dict) -> mo.ui.dropdown:
    """出自 preset (surrogate/*.yaml) の絞り込み dropdown。run_selector の上流フィルタ
    (preset を変えると出す run 群が変わる)。初期値は cfg (base.json⊕meta.json)。"""
    options = [ALL_PRESETS, *sorted(runs_df["preset"].dropna().unique())]
    return mo.ui.dropdown(
        options=options,
        value=valid_or(cfg.get("preset"), options, ALL_PRESETS),
        label="preset (yaml)",
    )


def preset_runs(runs_df: pd.DataFrame, preset_ui: mo.ui.dropdown) -> pd.DataFrame:
    """選択 preset の run だけに絞った runs_df (ALL_PRESETS なら素通し)。"""
    if preset_of(preset_ui) == ALL_PRESETS:
        return runs_df
    return cast(pd.DataFrame, runs_df[runs_df["preset"] == preset_of(preset_ui)])


def make_run_ui(
    runs_df: pd.DataFrame, preset_ui: mo.ui.dropdown, cfg: dict
) -> mo.ui.table:
    """run 選択テーブル。preset で絞り comp_type∈TARGET_MODEL の代表 run
    (sweep 親/単発 = parent_id 欠損) だけ出す。子は隠す。初期選択=cfg run_id。"""
    in_preset = preset_runs(runs_df, preset_ui)
    reps = in_preset[
        in_preset["comp_type"].isin(TARGET_MODEL) & in_preset["parent_id"].isna()
    ]
    runs = pd.DataFrame(reps[["tags.mlflow.runName", "comp_type", "run_id"]])
    wanted = set(cfg.get("sim", {}).get("run_selector") or [])
    ids = list(runs["run_id"])
    initial = [i for i, r in enumerate(ids) if r in wanted] or ([0] if ids else [])
    return mo.ui.table(
        runs, label="Run (1件)", selection="single", initial_selection=initial
    )


# ---------------------------------------------------------------------------
# Draw settings (表示調整のみ widget で残す)
# ---------------------------------------------------------------------------


def _comp_names(loaded_single: SurrogateBundle | None) -> list[str]:
    """draw の comp 選択肢 = 選択 run の代表 target (TARGET_MODEL[comp_type][0]、comp
    最豊富な正準モデル) の comp 名。run 未選択なら空。"""
    if loaded_single is None:
        return []
    return list(MCMODELS[TARGET_MODEL[comp_type_of(loaded_single.meta)][0]].names)


def make_draw_ui(loaded_single: SurrogateBundle | None, cfg: dict) -> mo.ui.dictionary:
    # draw_ui は 1 段フラット (ネストの益より深い添字アクセスの害が大きい)。sweep 系は
    # 条件付き (sweep 可能な電流のときだけ) で sweep_* キーを足す。
    names = _comp_names(loaded_single)
    default_comp = "soma" if "soma" in names else (names[0] if names else None)
    p = cfg.get("draw", {})
    d: dict = {
        # plt_style は描画設定なので draw_ui に置く (描画セルが setup_mpl で適用)。
        "plt_style": mo.ui.radio(options=STYLES, value=p.get("plt_style", STYLES[1])),
        # 既定=soma (全モデルが細胞体を "soma" と命名する共通規約)。
        "eval_comp": mo.ui.dropdown(
            options=names,
            value=valid_or(p.get("eval_comp"), names, default_comp),
            label="評価対象comp",
        ),
        # 全 comp を並べる図 (simple / train_*) の表示制限。空 = 全部 (既定)。
        # eval_comp (比較対象 1 件) とは別軸: traub19 の 19 本重ねを読める本数へ絞る。
        "view_comps": mo.ui.multiselect(
            options=names,
            value=[c for c in p.get("view_comps", []) if c in names],
            label="表示comp (空=全部)",
        ),
        "spike_orig": mo.ui.number(
            value=int(p.get("spike_orig", 0)), step=1, label="spike orig #"
        ),
        "spike_surr": mo.ui.number(
            value=int(p.get("spike_surr", 0)), step=1, label="spike surr #"
        ),
    }
    d.update(analysis_sweep.draw_fields(cfg, p) or {})
    return mo.ui.dictionary(d)


# ---------------------------------------------------------------------------
# View (計算済 save entry の合成・プレビュー)
# ---------------------------------------------------------------------------


def view_result(
    loaded_single: SurrogateBundle | None,
    loaded_sweep: list[SurrogateBundle],
    res_single: dict[str, EvalResult] | None,
    res_sweep: dict | None,
    draw: dict,
) -> dict[str, list[SaveEntry]]:
    """save entry を model / single / sweep の 3 グループに分ける (表示は
    panel.render_groups がタブ分け、保存は panel.flatten で一括)。model=静的モデル図
    (run ロードのみ)、single=波形+指標 (res_single ゲート)、sweep=掃引。"""
    return {
        "model": analysis_single.model_view(loaded_single, draw),
        "single": analysis_single.eval_view(loaded_single, res_single, draw),
        "sweep": analysis_sweep.view(loaded_sweep, res_sweep, draw),
    }


def plot_preview(cfg: dict) -> list[SaveEntry]:
    return current_preview_fig(**current_inputs(cfg))
