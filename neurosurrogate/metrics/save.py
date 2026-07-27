"""成果物 (図 / 表) の運搬と永続化。marimo 非依存。

view が返す `(id, fig)` 列に**保存名**を与えた `SaveEntry` が、表示と保存の共通の
運搬形になる (同じ列を UI は描画に、保存は書き出しに流す = 表示と保存が食い違わない)。
何を書けるか・どんな名前で・どう書き出すかは `SaveEntry` 1 つが持ち、UI 側は
「どれを選んだか」「どこへ」だけを渡す。
"""

import json
import re
from dataclasses import dataclass
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
    """1 成果物 = 表示名 + 中身 (図 or 表)。

    **保存名は表示名から決まる** (拡張子だけ中身の型で分かれる) ので別に持たない =
    表示と保存で名前が食い違わない。書き出し方も中身の型で決まるのでここが持つ。
    """

    name: str
    obj: Figure | pd.DataFrame

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


def save_entries(entries: list[SaveEntry], dest: Path, meta: dict) -> list[Path]:
    """entry を全部 `dest` 直下へ書き出し、`meta.json` (再現用の設定 snapshot) を
    同階層に置く。返り値は書いたパス列 (呼び出し側は表示に流すだけ)。"""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str)
    )
    return [e.write(dest) for e in entries]
