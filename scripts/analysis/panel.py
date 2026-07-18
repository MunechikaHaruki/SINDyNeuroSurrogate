from __future__ import annotations

import typing
from dataclasses import dataclass, field
from pathlib import Path

import marimo as mo
import pandas as pd
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "scripts" / "conf" / "surrogate" / "result"


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


def pair(base_ui: mo.ui.dictionary) -> str:
    """選択ペアのタグ文字列 (例 'hh→hh')。既定保存名に使う。"""
    train, target = base_ui["model_pair"].value
    return f"{train}→{target}"


# ---------------------------------------------------------------------------
# Panel (表示 blocks + 保存 entries)
# ---------------------------------------------------------------------------


@dataclass
class Panel:
    """表示 blocks と保存 entries を同時に積む。`figs` が「表示した fig をそのまま
    save entry に載せる」不変条件を1箇所に閉じ、view 関数を上から順に読める形に保つ。"""

    blocks: list[mo.Html] = field(default_factory=list)
    entries: list[SaveEntry] = field(default_factory=list)

    def note(self, md: str) -> None:
        """保存対象でない見出し/注記のみ追加。"""
        self.blocks.append(mo.md(md))

    def section(self, title: str, body: mo.Html) -> None:
        """### 見出し + 完成済み body (html/df) を表示。保存は別途 `save`。"""
        self.blocks += [mo.md(f"### {title}"), body]

    def figs(self, title: str, named: list[tuple[str, Figure]]) -> None:
        """### 見出し下に各 fig を表示し、同一 object を save entry へ登録。"""
        self.blocks.append(mo.md(f"### {title}"))
        for name, fig in named:
            self.blocks += [mo.md(f"##### {name}"), mo.mpl.interactive(fig)]
            self.entries.append(entry(name, fig))

    def save(self, name: str, obj: SaveItem) -> None:
        """表示済み (or 表示不要) の obj を save entry のみ登録。"""
        self.entries.append(entry(name, obj))

    def done(self) -> tuple[mo.Html, list[SaveEntry]]:
        return mo.vstack(self.blocks), self.entries


# ---------------------------------------------------------------------------
# Save Panel
# ---------------------------------------------------------------------------


SAVERS: dict[type, typing.Callable[[typing.Any, Path], None]] = {
    Figure: lambda o, p: o.savefig(p, dpi=300, bbox_inches="tight"),
    pd.DataFrame: lambda o, p: o.to_csv(p),
}


def make_save_panel(entries: list[SaveEntry]) -> mo.ui.dictionary:
    """SaveEntry から各 name の path入力＋保存ボタンを生成。"""
    return mo.ui.dictionary(
        {
            e.name: mo.ui.dictionary(
                {
                    "path": mo.ui.text(value=e.path, label=e.name),
                    "save": mo.ui.run_button(label="save"),
                }
            )
            for e in entries
        }
    )


def render_save_panel(panel: mo.ui.dictionary) -> mo.Html:
    rows = [
        mo.hstack(
            [item["path"], item["save"]],
            justify="start",
        )
        for item in panel.values()
    ]
    return mo.vstack([mo.md("### 画像保存パネル (docs/result/ 配下)"), *rows])


def save(save_panel: mo.ui.dictionary, entries: list[SaveEntry]) -> mo.Html:
    msgs: list[mo.Html] = []
    for e in entries:
        ctrl = save_panel[e.name]
        if not ctrl["save"].value:
            continue
        path = RESULT_DIR / str(ctrl["path"].value).strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        SAVERS[type(e.obj)](e.obj, path)
        msgs.append(mo.md(f"✅ {e.name}: `{path.relative_to(REPO_ROOT)}`"))
    return mo.vstack(msgs) if msgs else mo.md("(未保存)")
