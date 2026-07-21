from __future__ import annotations

import json
from pathlib import Path

import marimo as mo
from analysis.access import current_of

# ---------------------------------------------------------------------------
# 状態 snapshot / 復元 (meta.json)
#
# save 時に UI 値を「復元可能な形」で meta.json へ書き出し、dropdown 選択で
# 読み戻す。run_selector は DataFrame 非可逆のため run_id リストへ落として保持
# (raw .value dump は default=str で str 化され round-trip しない)。
# ---------------------------------------------------------------------------


def _run_ids(sub_ui: mo.ui.dictionary) -> list[str]:
    return sub_ui["run_selector"].value["run_id"].tolist()


def to_meta(
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
    draw_ui: mo.ui.dictionary,
) -> dict:
    """base/setting/draw UI 値を復元可能 snapshot に。make_*_ui の preset 引数と対。"""
    meta: dict = {
        "base": {
            "plt_style": base_ui["plt_style"].value,
            "sim_current_type": current_of(base_ui),
            "dt": base_ui["dt"].value,
            "model_pair": list(base_ui["model_pair"].value),
        },
        "sim": {
            "run_ids": _run_ids(setting_ui["sim"]),
            "current_params": setting_ui["sim"]["current_params"].value or {},
        },
        "draw": {
            "eval_comp": draw_ui["eval_comp"].value,
            "spike": {
                "orig": draw_ui["single"]["spike"]["orig"].value,
                "surr": draw_ui["single"]["spike"]["surr"].value,
            },
        },
    }
    if "sweep" in setting_ui:
        sweep_ui = setting_ui["sweep"]
        meta["sweep"] = {
            "run_ids": _run_ids(sweep_ui),
            "amp_start": sweep_ui["amp_start"].value,
            "amp_stop": sweep_ui["amp_stop"].value,
            "amp_steps": sweep_ui["amp_steps"].value,
        }
    if "sweep" in draw_ui:
        sweep_draw = draw_ui["sweep"]
        meta["draw"]["sweep"] = {
            "metric": sweep_draw["metric"].value,
            "ylim": {
                "auto": sweep_draw["ylim"]["auto"].value,
                "min": sweep_draw["ylim"]["min"].value,
                "max": sweep_draw["ylim"]["max"].value,
            },
        }
    return meta


def _list_metas(result_dir: Path) -> dict[str, str]:
    """result_dir 直下の保存 dir (single/sweep) の meta.json を走査し label→path。"""
    return {
        str(p.parent.relative_to(result_dir)): str(p)
        for p in sorted(result_dir.glob("*/meta.json"))
    }


def make_panel(result_dir: Path) -> tuple[mo.Html, mo.ui.dropdown]:
    """復元パネル (html, dropdown) を返す。dropdown 選択で即復元・空選択で既定。
    run_button は click 後 False 復帰し gate が revert するため不採用。"""
    dropdown = mo.ui.dropdown(options=_list_metas(result_dir), label="復元元 meta.json")
    html = mo.vstack([mo.md("### 状態復元 (meta.json) — 選択で即復元"), dropdown])
    return html, dropdown


def load(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text())
