"""成果物 (図 / 表) の運搬と永続化。marimo 非依存。

view が返す `(id, fig)` 列に**保存名**を与えた `SaveEntry` が、表示と保存の共通の
運搬形になる (同じ列を UI は描画に、保存は書き出しに流す = 表示と保存が食い違わない)。
何を書けるか・どんな名前で・どう書き出すかは `SaveEntry` 1 つが持ち、UI 側は
「どれを選んだか」「どこへ」だけを渡す。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

_UNSAFE = re.compile(r"[\s/\\:]+")


def slug(name: str) -> str:
    """図 id の 1 区切りに使う名前をパス安全へ。run 軸キー (`meta.label`) は凡例で
    折り返すための改行や `/` を含むので、そのまま名前に混ぜると保存時に階層が
    割れる (表示名 = 保存名の規約を保ったまま名前側だけ潰す)。"""
    return _UNSAFE.sub("-", name.strip())


@dataclass(frozen=True)
class SaveEntry:
    """1 成果物 = 表示名 + 中身 (図 or 表) + 由来 (参照した評価 run と描画設定)。

    **保存名は表示名から決まる** (拡張子だけ中身の型で分かれる) ので別に持たない =
    表示と保存で名前が食い違わない。書き出し方も中身の型で決まるのでここが持つ。
    `sources`/`draw` は `meta.json` の対応する value にそのまま落ちる = 「どの
    リソースからどう描いたか」を成果物 1 件ごとに追跡できる。`draw` はどの表示設定
    (`report.spec` の `DrawSpec`/`CompareSpec`) でも中身を見ず `is_dataclass` で
    しか判定しない
    (dict 化は `save_entries` が meta.json へ書き出す境界でだけ行う) → 具体型への
    依存を持たない。
    """

    name: str
    obj: Figure | pd.DataFrame
    sources: tuple[str, ...] = ()  # 参照した評価 run の id (由来なしは空)
    draw: object | None = None  # 使った表示設定 dataclass (無ければ None)

    @property
    def path(self) -> str:
        """保存先ディレクトリからの相対パス。name の `/` (`<評価>/<run>/<図名>` の
        区切り) はそのままディレクトリ階層になる。"""
        return f"{self.name}{'.csv' if isinstance(self.obj, pd.DataFrame) else '.png'}"

    def write(self, dest: Path) -> Path:
        path = dest / self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(self.obj, pd.DataFrame):
            self.obj.to_csv(path)
        else:
            self.obj.savefig(path, dpi=300, bbox_inches="tight")
        return path


def _entries_meta(entries: list[SaveEntry]) -> dict:
    """entry 列 → `meta.json` のスキーマ (`保存パス → {sources, draw}` の対応表。
    描画宣言の丸ごと保存ではなく成果物 1 件ごとの由来)。副作用なしの純粋関数で
    書き出し (`save_entries`) と分離し、スキーマ組立だけを単独でテストできる。"""
    return {
        e.path: {
            "sources": list(e.sources),
            "draw": asdict(e.draw)
            if is_dataclass(e.draw) and not isinstance(e.draw, type)
            else None,
        }
        for e in entries
    }


def save_entries(entries: list[SaveEntry], dest: Path) -> list[Path]:
    """entry を全部 `dest` 直下へ書き出し、`_entries_meta` が組んだスキーマを
    `meta.json` として同階層に置く。返り値は書いたパス列 (呼び出し側は表示に
    流すだけ)。"""
    dest.mkdir(parents=True, exist_ok=True)
    meta = _entries_meta(entries)
    (dest / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str)
    )
    return [e.write(dest) for e in entries]
