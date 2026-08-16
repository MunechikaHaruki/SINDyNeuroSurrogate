"""成果物 (図 / 表) を `results/` へ落とす層 = **保存段の名前と書き出しの唯一の
置き場所**。

描画ドメイン (`neurosurrogate.report`) が返すのは名前と中身だけ
(`plotting.Artifact`) で、
それを **MLflow の 3 experiment がそのまま 3 段**の名前 (`models/<学習 run>/`,
`series/<評価 run>/`, `report/<レポート run>/`) に変えるのは各 experiment の module
(`surrogate` / `series` / `report`)。**どの描画関数を呼んだかが段を決める**ので、
図の側に段の種別を持たせない。

ここが持つのはその 3 module に共通する部分だけ: run 名 (`run_name`)、段 1 つの
組み立て (`stage`)、運搬形 (`SaveEntry`)、書き出し (`save_entries`)。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import mlflow
import pandas as pd

from neurosurrogate.plotting import Artifact

from . import logger

_UNSAFE = re.compile(r"[\s/\\:]+")


def slug(name: str) -> str:
    """図 id の 1 区切りに使う名前をパス安全へ。run 軸キー (`meta.label`) は凡例で
    折り返すための改行や `/` を含み、保存段の名前 (MLflow の run 名) は人が付け替え
    られるので、そのまま名前に混ぜると保存時に階層が割れる (表示名 = 保存名の規約を
    保ったまま名前側だけ潰す)。

    空になる名前と `.` / `..` も潰す — **1 段は必ず 1 段**でなければ、段が消えたり
    上の階層へ抜けたりして「run 1 つ = ディレクトリ 1 つ」の対応が崩れる。
    """
    out = _UNSAFE.sub("-", name.strip())
    return "-" if out in ("", ".", "..") else out


def run_name(run_id: str) -> str:
    """MLflow の run 名。**保存段の名前** = MLflow UI の run 名なので、ディレクトリから
    UI 側の run をそのまま引ける。引けない run (消された等) は id を段名にして描画は
    通す (名前が取れないことは図を出さない理由にならない)。

    学習 run にも評価 run にもレポート run にも効く (run 名は experiment を問わない
    属性)。"""
    try:
        return mlflow.get_run(run_id).info.run_name or run_id
    except Exception as e:
        logger.debug(f"run {run_id} の名前解決に失敗: {e}")
        return run_id


def stage(kind_dir: str, run_id: str) -> str:
    """保存段 1 つ = `<段>/<run 名>-<run id 先頭>`。

    **run id を混ぜるのは名前が同一性でないから**: MLflow の run 名は人が付け替え
    られて一意でなく (掃引違いの評価 run は同名になる)、`slug` も単射でない
    (`a b` と `a/b` は同じ)。名前だけを段にすると別 run の図が後勝ちで潰し合う。

    段は必ず run に紐づく (保存する成果物はどれも MLflow から読んだ run が由来) =
    id 無しの経路は持たない。"""
    return f"{kind_dir}/{slug(run_name(run_id))}-{run_id[:8]}"


@dataclass(frozen=True)
class SaveEntry:
    """描画層の成果物 1 件 (`Artifact`) に**保存の関心だけ**を足したもの: どの段へ
    置くか (`stage`) と、由来 (`sources` / `draw`)。

    成果物そのものは包み直さず `Artifact` を持つ (同じ「名前 + 中身」を層ごとに
    別の型へ写し替えない)。**保存名は段 + 成果物の名前から決まる** (拡張子だけ中身の
    型で分かれる) ので別に持たない = 表示と保存で名前が食い違わない。書き出し方も
    中身の型で決まるのでここが持つ。

    `sources`/`draw` は `meta.json` の対応する value にそのまま落ちる = 「どの
    リソースからどう描いたか」を成果物 1 件ごとに追跡できる。`draw` はどの表示設定
    (`tuning.Tuning` など) でも中身を見ず `is_dataclass` でしか判定しない (dict 化は
    `save_entries` が meta.json へ書き出す境界でだけ行う) → 具体型への依存を持たない。
    """

    stage: str  # 保存段 (`<experiment の段>/<run 名>-<run id 先頭>`)
    artifact: Artifact
    sources: tuple[str, ...] = ()  # 描くのに読んだ run の id (由来なしは空)
    draw: object | None = None  # 使った表示設定 dataclass (無ければ None)

    @property
    def path(self) -> str:
        """保存先ディレクトリからの相対パス。段と成果物の名前の `/` はそのまま
        ディレクトリ階層になる。"""
        ext = ".csv" if isinstance(self.artifact.obj, pd.DataFrame) else ".png"
        return f"{self.stage}/{self.artifact.name}{ext}"

    def write(self, dest: Path) -> Path:
        path = dest / self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(self.artifact.obj, pd.DataFrame):
            self.artifact.obj.to_csv(path)
        else:
            self.artifact.obj.savefig(path, dpi=300, bbox_inches="tight")
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


def _read_meta(dest: Path) -> dict:
    """既存の `meta.json` (無ければ空)。手で壊れた JSON を置いた場合も描画は通す
    (由来の記録は成果物の付帯情報で、書き出しを止める理由にしない)。"""
    path = dest / "meta.json"
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def save_entries(entries: list[SaveEntry], dest: Path) -> list[Path]:
    """entry を全部 `dest` 直下へ書き出し、`_entries_meta` が組んだスキーマを
    `meta.json` として同階層に置く。返り値は書いたパス列 (呼び出し側は表示に
    流すだけ)。

    既存の `meta.json` には**上書きでなく合流**する。キーが保存パスなので合流の
    意味が一意に決まり、同じ `dest` に別の系列を描き足しても前の由来が消えない
    (レポートごとに dest を割らない代わりの担保)。
    """
    dest.mkdir(parents=True, exist_ok=True)
    # 書き出しが先で、途中で落ちても `finally` で**書けた分だけ**の由来を残す
    # (meta.json が無いファイルを主張しない / 書けたのに由来が消えない の両立)。
    done: list[SaveEntry] = []
    try:
        for e in entries:
            e.write(dest)
            done.append(e)
    finally:
        (dest / "meta.json").write_text(
            json.dumps(
                _read_meta(dest) | _entries_meta(done),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    return [dest / e.path for e in done]
