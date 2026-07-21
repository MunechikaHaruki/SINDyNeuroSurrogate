from __future__ import annotations

import json
import typing
from dataclasses import dataclass
from pathlib import Path

import marimo as mo
import pandas as pd
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


def make_save_panel(groups: dict[str, list[SaveEntry]]) -> mo.ui.dictionary:
    """グループ (current/result) ごとに「対象 entry の複数選択 + 保存ボタン」を生成。

    multiselect 既定は全選択 (従来の一括保存と同挙動)。選択を外した entry は保存対象外。
    """
    return mo.ui.dictionary(
        {
            name: mo.ui.dictionary(
                {
                    "select": mo.ui.multiselect(
                        options=[e.name for e in entries],
                        value=[e.name for e in entries],
                        label="対象",
                    ),
                    "run": mo.ui.run_button(label=f"save {name}"),
                }
            )
            for name, entries in groups.items()
        }
    )


def make_save_dirs(groups: dict[str, list[SaveEntry]]) -> mo.ui.dictionary:
    """current 以外の group ごとに保存先ディレクトリ入力を生成。

    single/sweep を個別指定する。
    """
    return mo.ui.dictionary(
        {
            name: mo.ui.text(value=f"_{name}", label=f"{name} 保存先")
            for name in groups
            if name != "current"
        }
    )


def render_save_panel(panel: mo.ui.dictionary, save_dirs: mo.ui.dictionary) -> mo.Html:
    rows = []
    for name, ctrl in panel.items():
        parts = [save_dirs[name]] if name in save_dirs else []
        parts += [ctrl["select"], ctrl["run"]]
        rows.append(mo.vstack(parts))
    return mo.vstack([mo.md("### 画像保存パネル"), *rows])


def _dest(name: str, result_dir: Path, save_dirs: dict[str, str]) -> Path:
    """保存先ルート。

    dir 入力を持つ group (single/sweep) は `result_dir/<入力>/` 直下
    (fig と meta.json を同階層)。持たない current は従来どおり `result_dir/current/`。
    """
    if name in save_dirs:
        return result_dir / save_dirs[name]
    return result_dir / name


def save(
    save_panel: mo.ui.dictionary,
    groups: dict[str, list[SaveEntry]],
    result_dir: Path,
    save_dirs: dict[str, str],
    meta: dict,
) -> mo.Html:
    """押されたグループを一括保存。

    dir 入力を持つ group (single/sweep) は入力ディレクトリ直下に fig と `meta.json`
    (base/setting/draw UI の値) を同階層で置く。
    """
    msgs: list[mo.Html] = []
    for name, entries in groups.items():
        ctrl = save_panel[name]
        if not ctrl["run"].value:
            continue
        selected = set(ctrl["select"].value)
        dest = _dest(name, result_dir, save_dirs)
        dest.mkdir(parents=True, exist_ok=True)
        if name in save_dirs:
            (dest / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False, default=str)
            )
        for e in entries:
            if e.name not in selected:
                continue
            SAVERS[type(e.obj)](e.obj, dest / e.path)
            msgs.append(
                mo.md(f"✅ {e.name}: `{(dest / e.path).relative_to(result_dir)}`")
            )
    return mo.vstack(msgs) if msgs else mo.md("(未保存)")
