from __future__ import annotations

from typing import cast

import marimo as mo
import pandas as pd

# ---------------------------------------------------------------------------
# base_ui/setting_ui/draw_ui read 規約の集約 (leaf: marimo のみ依存)。
#
# ui/mode が共有する UI dict の掘削をここへ一元化。ui.py に置くと
# ui→mode の import と衝突し mode 側が使えない (循環) ため独立 module に切出す。
# ---------------------------------------------------------------------------


def target_of(base_ui: mo.ui.dictionary) -> str:
    """適用先 MC モデル名 (model_pair の target)。"""
    return str(base_ui["model_pair"].value[1])


ALL_PRESETS = "(すべて)"  # preset dropdown の「絞らない」選択肢


def preset_of(preset_ui: mo.ui.dropdown) -> str:
    """選択中の preset (surrogate/*.yaml)。ALL_PRESETS なら絞り込まない。

    base_ui の外に置く: preset を変えると model_pair の選択肢自体が変わる (整合的な
    ペアだけを出す) ため、base_ui より **上流**の独立 UI でなければならない
    (marimo は自分自身に依存するセルを再実行できない)。
    """
    return str(preset_ui.value)


def comp_type_of(base_ui: mo.ui.dictionary) -> str:
    """置換対象のコンパートメント種類名 (model_pair の左)。"""
    return str(base_ui["model_pair"].value[0])


def current_of(base_ui: mo.ui.dictionary) -> str:
    """選択電流タイプ名。"""
    return str(base_ui["sim_current_type"].value)


def dt_of(base_ui: mo.ui.dictionary) -> float:
    """シミュ刻み幅 dt。"""
    return float(base_ui["dt"].value)


def plt_style_of(base_ui: mo.ui.dictionary) -> str:
    """matplotlib style 名。"""
    return str(base_ui["plt_style"].value)


def eval_comp_of(draw_ui: mo.ui.dictionary) -> str:
    """評価対象 comp 名。"""
    return str(draw_ui["eval_comp"].value)


def sim_current_params_of(setting_ui: mo.ui.dictionary) -> dict:
    """single モード current_params 値。"""
    return setting_ui["sim"]["current_params"].value or {}


def sim_run_selection_of(setting_ui: mo.ui.dictionary) -> pd.DataFrame:
    """single モード選択 run の DataFrame (run_id 列を含む)。"""
    return cast(pd.DataFrame, setting_ui["sim"]["run_selector"].value)


def sweep_run_selection_of(setting_ui: mo.ui.dictionary) -> pd.DataFrame | None:
    """sweep モード選択 run の DataFrame。sweep UI 非対応時は None。"""
    if "sweep" not in setting_ui:
        return None
    return cast(pd.DataFrame, setting_ui["sweep"]["run_selector"].value)


def valid_or(value: object, options: object, default: object) -> object:
    """preset 復元値が現 options に含まれれば採用、無ければ default。
    run 集合変化などで無効化した選択 (dropdown 値) を既定へ吸収する共通規約。"""
    return value if value in options else default  # type: ignore[operator]
