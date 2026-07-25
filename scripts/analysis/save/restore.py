from __future__ import annotations

import json
from pathlib import Path

import marimo as mo
import pandas as pd

# ---------------------------------------------------------------------------
# 設定の解決 (base.json デフォルト ← meta.json 上書き) と snapshot / 復元
#
# marimo は widget を減らし「run 選択 + 実行 + 表示」に専念する。シミュ入力
# (current_type/dt/current_params/sweep 範囲) は widget でなく **設定 dict cfg** から
# 読む: conf/base.json のデフォルトを、復元 dropdown で選んだ保存 meta.json が上書き
# する (meta が勝つ)。保存時は使用した cfg を meta.json に書き戻すので round-trip する。
# comp_type は cfg に持たない (選択 run の SurrogateMeta から自動決定)。
# ---------------------------------------------------------------------------

BASE_JSON = Path(__file__).resolve().parents[2] / "conf" / "base.json"


def load_base() -> dict:
    """デフォルト設定 (conf/base.json)。"""
    return json.loads(BASE_JSON.read_text())


def _merge(base: object, override: object) -> object:
    """deep merge: 双方 dict なら key ごとに再帰、それ以外は override 採用
    (list/scalar は丸ごと差し替え)。"""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(base.get(k), v) if k in base else v
    return out


def resolve(meta_path: str | None) -> dict:
    """base.json デフォルト ← 選択 meta.json 上書き。meta 未選択なら base のみ。"""
    if not meta_path:
        return load_base()
    return _merge(load_base(), json.loads(Path(meta_path).read_text()))


def _snapshot(value: object) -> object:
    """UI .value を復元可能形へ。run_selector の DataFrame → run_id リスト。"""
    if isinstance(value, pd.DataFrame):
        return value["run_id"].tolist()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    return value


def to_meta(
    preset_ui: mo.ui.dropdown,
    cfg: dict,
    run_selector: mo.ui.table,
    draw: dict,
) -> dict:
    """使用した設定を復元可能 snapshot に (base.json と同じ形)。cfg (base⊕meta の
    解決値) をベースに、実際に効いた preset / run 選択 / draw 値で上書き。draw は
    marimo で .value 済みの dict。sim.current_params と sweep 範囲は cfg から素通し。"""
    return {
        **cfg,
        "preset": preset_ui.value,
        "sim": {**cfg.get("sim", {}), "run_selector": _snapshot(run_selector.value)},
        "draw": _snapshot(draw),
    }


def _list_metas(result_dir: Path) -> dict[str, str]:
    """result_dir 直下の保存 dir (single/sweep) の meta.json を走査し label→path。"""
    return {
        str(p.parent.relative_to(result_dir)): str(p)
        for p in sorted(result_dir.glob("*/meta.json"))
    }


def make_panel(result_dir: Path) -> tuple[mo.Html, mo.ui.dropdown]:
    """復元パネル (html, dropdown)。dropdown 選択で即復元・空選択で既定 (base.json
    のみ)。run_button は click 後 False 復帰し gate が revert するため不採用。"""
    dropdown = mo.ui.dropdown(options=_list_metas(result_dir), label="復元元 meta.json")
    html = mo.vstack(
        [mo.md("### 設定復元 (meta.json) — 選択で base.json を上書き"), dropdown]
    )
    return html, dropdown
