from __future__ import annotations

import typing
from pathlib import Path
from typing import Literal

import marimo as mo
import matplotlib.pyplot as plt
import pandas as pd
from analysis.access import (
    comp_type_of,
    current_of,
    dt_of,
    sim_current_params_of,
    target_of,
    valid_or,
)
from analysis.mode import single as analysis_single
from analysis.mode import sweep as analysis_sweep
from analysis.save.panel import SaveEntry
from mlflow_io import setup_mlflow

from neurosurrogate.currents import CURRENT_MAP
from neurosurrogate.metrics.eval import EvalResult
from neurosurrogate.models import MCMODELS
from neurosurrogate.surrogate.bundle import SurrogateBundle
from neurosurrogate.surrogate.replace import replaced_names
from neurosurrogate.view.utils import current_preview_fig

CurrentList: list = list(CURRENT_MAP.keys())
MplStyle = Literal["paper", "presentation"]

setup_mlflow()


# ---------------------------------------------------------------------------
# Base UI
# ---------------------------------------------------------------------------


def make_base_ui(
    runs_df: pd.DataFrame,
    target_model: dict[str, list[str]],
    preset: dict | None = None,
) -> mo.ui.dictionary:
    # モデルペア = **置換対象のコンパートメント種類 → 適用先 MC モデル**。サロゲート
    # は「種類 → それを置換するモデル」の対応 (replace.replaceable) なので、左は学習
    # データの MC モデル名ではなく comp_type を取る。右は target_model が種類ごとに
    # 定義する適用先一覧。label→(comp_type,target) を .value で得る。
    comp_types = sorted(runs_df["comp_type"].unique())
    pairs = {
        f"{comp_type}→{tgt}": (comp_type, tgt)
        for comp_type in comp_types
        for tgt in target_model.get(comp_type, [])
    }
    # ペアが 1 つも組めない = 選べる run が無い。空 dropdown を作って下流を
    # StopIteration/KeyError で落とさず、原因 (run 側か TARGET_MODEL 側か) を示す。
    if not pairs:
        raise ValueError(
            "選択可能なモデルペアが無い。"
            f"読込めた run の comp_type={comp_types or '(なし)'} / "
            f"TARGET_MODEL のキー={list(target_model)}。"
            "run が 0 件なら surrogate の pickle スキーマ変更で旧 run が読めていない "
            "(再学習が要る)。comp_type が TARGET_MODEL に無いなら適用先を定義する。"
        )
    plt_options = list(typing.get_args(MplStyle))
    # preset (復元) 値で初期値上書き。無効値 (run 集合変化等) は既定へフォールバック。
    # model_pair は json で list 化するので options(tuple) と list 比較で照合。
    b = (preset or {}).get("base", {})
    pair_label = next(
        (k for k, v in pairs.items() if list(v) == b.get("model_pair")),
        next(iter(pairs)),
    )
    return mo.ui.dictionary(
        {
            "plt_style": mo.ui.radio(
                options=plt_options, value=b.get("plt_style", plt_options[1])
            ),
            "sim_current_type": mo.ui.dropdown(
                CurrentList,
                value=valid_or(b.get("sim_current_type"), CurrentList, "lin&steady"),
            ),
            "dt": mo.ui.number(value=b.get("dt", 0.01), step=0.001),
            "model_pair": mo.ui.dropdown(
                options=pairs,
                value=pair_label,
                label="モデルペア (train→target)",
            ),
        }
    )


# ---------------------------------------------------------------------------
# Setting UI (集約)
# ---------------------------------------------------------------------------


def setup_mpl(matplotlib_style: str):
    style_dir = Path(__file__).resolve().parents[1] / "conf" / "style"
    plt.style.use(style_dir / "base.mplstyle")
    plt.style.use(style_dir / f"{matplotlib_style}.mplstyle")


def _run_selector(
    runs: pd.DataFrame,
    label: str,
    selection: Literal["single", "multi"] = "multi",
    selected_ids: list[str] | None = None,
) -> mo.ui.table:
    # preset 復元時は run_id → 表示行位置へ写像。無指定は既定 (先頭 1 件)。
    ids = list(runs["run_id"])
    if selected_ids is not None:
        wanted = set(selected_ids)
        initial = [i for i, r in enumerate(ids) if r in wanted]
    else:
        initial = [0] if ids else []
    return mo.ui.table(
        runs, label=label, selection=selection, initial_selection=initial
    )


def make_setting_ui(
    runs_df: pd.DataFrame,
    base_ui: mo.ui.dictionary,
    preset: dict | None = None,
) -> mo.ui.dictionary:
    current_type = current_of(base_ui)
    # 選んだペア (種類 → 適用先) に**実際に置換できる** run だけを提示。互換基準は
    # replace ドメインの判定 (種類一致 + params 両立) をそのまま使い、UI 側に複製
    # しない。run_selector は sim/sweep 各キーへ個別に埋め、single (1件必須) と
    # sweep (複数可) で選択状態を分離する。
    net = MCMODELS[target_of(base_ui)]
    runs = pd.DataFrame(
        runs_df[
            (runs_df["comp_type"] == comp_type_of(base_ui))
            & runs_df["meta"].map(lambda m: bool(replaced_names(m, net)))
        ][["tags.mlflow.runName", "run_id"]]
    )
    sim_p = (preset or {}).get("sim", {})
    sweep_p = (preset or {}).get("sweep", {})
    d: dict = {
        "sim": analysis_single.make_sim_ui(
            current_type,
            _run_selector(
                runs, "single Run (1件)", "single", sim_p.get("run_selector")
            ),
            sim_p.get("current_params"),
        ),
        "run_sim": mo.ui.run_button(label="single 実行"),
    }
    sweep = analysis_sweep.make_sweep_ui(
        current_type,
        _run_selector(
            runs, "sweep Run (複数可)", selected_ids=sweep_p.get("run_selector")
        ),
        sweep_p,
    )
    if sweep is not None:
        d["sweep"] = sweep
        d["run_sweep"] = mo.ui.run_button(label="sweep 実行")
    return mo.ui.dictionary(d)


# ---------------------------------------------------------------------------
# Draw setttings
# ---------------------------------------------------------------------------


def make_draw_ui(
    base_ui: mo.ui.dictionary, preset: dict | None = None
) -> mo.ui.dictionary:
    net = MCMODELS[target_of(base_ui)]
    p = (preset or {}).get("draw", {})
    d: dict = {
        # 既定=soma (全モデルが細胞体を "soma" と命名する共通規約)。
        "eval_comp": mo.ui.dropdown(
            options=net.names,
            value=valid_or(p.get("eval_comp"), net.names, "soma"),
            label="評価対象comp",
        ),
        "single": analysis_single.make_draw_ui(p.get("single", {}).get("spike")),
    }
    sweep = analysis_sweep.make_draw_ui(base_ui, p.get("sweep"))
    if sweep is not None:
        d["sweep"] = sweep
    return mo.ui.dictionary(d)


# ---------------------------------------------------------------------------
# View (計算済 save entry の合成・プレビュー)
# ---------------------------------------------------------------------------


def view_result(
    loaded_single: SurrogateBundle | None,
    loaded_sweep: list[SurrogateBundle],
    base_ui: mo.ui.dictionary,
    res_single: EvalResult | None,
    res_sweep: dict | None,
    draw_ui: mo.ui.dictionary,
) -> list[SaveEntry]:
    """single / sweep の save entry 列を連結 (表示は panel.render)。"""
    return analysis_single.view(
        loaded_single, base_ui, res_single, draw_ui
    ) + analysis_sweep.view(loaded_sweep, res_sweep, draw_ui)


def plot_preview(
    base_ui: mo.ui.dictionary, setting_ui: mo.ui.dictionary
) -> list[SaveEntry]:
    current_type = current_of(base_ui)
    return current_preview_fig(
        current_type,
        dt_of(base_ui),
        sim_current_params_of(setting_ui),
    )
