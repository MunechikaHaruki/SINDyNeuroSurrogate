from __future__ import annotations

import json
from pathlib import Path

import marimo as mo
import pandas as pd

# ---------------------------------------------------------------------------
# 状態 snapshot / 復元 (meta.json)
#
# UI 値 (.value) は make_*_ui の preset 引数と同じ木構造なので丸ごと dump するだけで
# 復元可能。唯一 run_selector の DataFrame だけ非可逆なので run_id リストへ落とす
# (他の tuple/scalar/dict は json で round-trip)。make_*_ui 側が同じ key を読む。
# ---------------------------------------------------------------------------


def _snapshot(value: object) -> object:
    """UI .value を復元可能形へ。run_selector の DataFrame → run_id リスト。"""
    if isinstance(value, pd.DataFrame):
        return value["run_id"].tolist()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    return value


def to_meta(
    preset_ui: mo.ui.dropdown,
    base_ui: mo.ui.dictionary,
    setting_ui: mo.ui.dictionary,
    draw_ui: mo.ui.dictionary,
) -> dict:
    """preset/base/sim/sweep/draw UI 値を復元可能 snapshot に (make_*_ui preset と対)。

    preset (yaml 絞り込み) は base_ui の外にある独立 UI なので個別に積む — 復元時は
    これが先に効き、整合する model_pair / run 一覧が組まれる。
    """
    meta: dict = {
        "preset": preset_ui.value,
        "base": _snapshot(base_ui.value),
        "sim": _snapshot(setting_ui["sim"].value),
        "draw": _snapshot(draw_ui.value),
    }
    if "sweep" in setting_ui:
        meta["sweep"] = _snapshot(setting_ui["sweep"].value)
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
