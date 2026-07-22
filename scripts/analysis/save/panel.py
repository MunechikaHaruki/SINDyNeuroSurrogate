from __future__ import annotations

import json
import typing
from dataclasses import dataclass
from pathlib import Path

import marimo as mo
import pandas as pd
from analysis.access import ALL_PRESETS
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Save Entry
# ---------------------------------------------------------------------------


SaveItem = Figure | pd.DataFrame


@dataclass(frozen=True)
class SaveEntry:
    name: str
    obj: SaveItem
    path: str  # default path (docs/slide/result 相対)


def entry(name: str, obj: SaveItem) -> SaveEntry:
    """name をそのまま既定ファイル名に (拡張子のみ付与)。呼び出し側が pair 等を含む
    最終的な表示名を組む。"""
    ext = ".csv" if isinstance(obj, pd.DataFrame) else ".png"
    return SaveEntry(name, obj, f"{name}{ext}")


# ---------------------------------------------------------------------------
# Render (save entry 列 → 表示。fig→interactive / df→table, 見出し=entry 名)
# ---------------------------------------------------------------------------


def render(entries: list[SaveEntry]) -> mo.Html:
    """save 対象をそのまま表示に流す (display と save の単一源)。"""
    blocks: list[mo.Html] = []
    for e in entries:
        body = (
            mo.mpl.interactive(e.obj)
            if isinstance(e.obj, Figure)
            else mo.ui.table(e.obj)
        )
        blocks += [mo.md(f"##### {e.name}"), body]
    return mo.vstack(blocks)


# ---------------------------------------------------------------------------
# Save Panel
# ---------------------------------------------------------------------------


SAVERS: dict[type, typing.Callable[[typing.Any, Path], None]] = {
    Figure: lambda o, p: o.savefig(p, dpi=300, bbox_inches="tight"),
    pd.DataFrame: lambda o, p: o.to_csv(p),
}


def _default_dir(preset: str) -> str:
    """保存先の既定名。preset で絞っていれば yaml 名を冠して実験群ごとに分ける
    (絞っていないときは冠する名前が無いので従来どおり)。"""
    return "_result" if preset == ALL_PRESETS else f"{preset}_result"


def make_save_panel(entries: list[SaveEntry], preset: str) -> mo.ui.dictionary:
    """result entry の「保存先 + 対象複数選択 + 保存ボタン」を生成。

    保存先の既定名は出自 preset (surrogate/*.yaml) 入り — 実験群ごとに別ディレクトリ
    へ落ち、後から「どの設定の図か」が名前だけで分かる。
    multiselect 既定は全選択 (従来の一括保存と同挙動)。選択を外した entry は保存対象外。
    """
    return mo.ui.dictionary(
        {
            "dir": mo.ui.text(value=_default_dir(preset), label="保存先"),
            "select": mo.ui.multiselect(
                options=[e.name for e in entries],
                value=[e.name for e in entries],
                label="対象",
            ),
            "run": mo.ui.run_button(label="save"),
        }
    )


def render_save_panel(panel: mo.ui.dictionary) -> mo.Html:
    return mo.vstack(
        [mo.md("### 画像保存パネル"), panel["dir"], panel["select"], panel["run"]]
    )


def save(
    save_panel: mo.ui.dictionary,
    entries: list[SaveEntry],
    result_dir: Path,
    meta: dict,
) -> mo.Html:
    """選択 entry を `result_dir/<入力>/` 直下へ保存 (fig と `meta.json` を同階層)。"""
    if not save_panel["run"].value:
        return mo.md("(未保存)")
    selected = set(save_panel["select"].value)
    dest = result_dir / save_panel["dir"].value
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str)
    )
    msgs: list[mo.Html] = []
    for e in entries:
        if e.name not in selected:
            continue
        SAVERS[type(e.obj)](e.obj, dest / e.path)
        msgs.append(mo.md(f"✅ {e.name}: `{(dest / e.path).relative_to(result_dir)}`"))
    return mo.vstack(msgs) if msgs else mo.md("(未保存)")
