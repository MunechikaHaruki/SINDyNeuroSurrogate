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


def render_groups(groups: dict[str, list[SaveEntry]]) -> mo.Html:
    """グループ (model/single/sweep) をタブ分け表示 (全部を 1 列に流すと見にくい)。
    空グループはタブごと省く。"""
    tabs = {name: render(es) for name, es in groups.items() if es}
    return mo.ui.tabs(tabs) if tabs else mo.md("(結果なし)")


def flatten(groups: dict[str, list[SaveEntry]]) -> list[SaveEntry]:
    """保存パネル用にグループを平坦化 (表示はグループ分割・保存は一括)。"""
    return [e for es in groups.values() for e in es]


# ---------------------------------------------------------------------------
# Save Panel
# ---------------------------------------------------------------------------


SAVERS: dict[type, typing.Callable[[typing.Any, Path], None]] = {
    Figure: lambda o, p: o.savefig(p, dpi=300, bbox_inches="tight"),
    pd.DataFrame: lambda o, p: o.to_csv(p),
}


def _default_dir(run_name: str | None) -> str:
    """保存先の既定名 = 選択 run の runName 冠 — どの run の図か名前だけで分かる。
    run 未選択なら冠する名前が無いので従来どおり。"""
    return f"{run_name}_result" if run_name else "_result"


def make_save_panel(entries: list[SaveEntry], run_name: str | None) -> mo.ui.dictionary:
    """result entry の「保存先 + 対象複数選択 + 保存ボタン」を生成。

    保存先の既定名は選択 run の runName 入り — run ごとに別ディレクトリへ落ち、後から
    「どの run の図か」が名前だけで分かる。
    multiselect 既定は全選択 (従来の一括保存と同挙動)。選択を外した entry は保存対象外。
    """
    return mo.ui.dictionary(
        {
            "dir": mo.ui.text(value=_default_dir(run_name), label="保存先"),
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
