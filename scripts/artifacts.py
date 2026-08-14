"""成果物 (図 / 表) を `results/` へ落とす層 = **保存段の名前と書き出しの唯一の
置き場所**。描画ドメイン (`neurosurrogate.report.figures`) は「どの run について
描いたか」しか名乗らず、それを **MLflow の 3 experiment がそのまま 3 段**の名前
(`models/<学習 run>/`, `series/<評価 run>/`, `report/<レポート run>/`) に変えるのが
ここ。run 名を解けるのは MLflow を知る側だけなので、ドメインには置けない。

`SaveEntry` が表示と保存の共通の運搬形 (同じ列を UI は描画に、保存は書き出しに流す
= 表示と保存が食い違わない)。何を書けるか・どんな名前で・どう書き出すかは
`SaveEntry` 1 つが持ち、UI 側は「どれを選んだか」「どこへ」だけを渡す。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from neurosurrogate.report.figures import (
    MODEL,
    ORIGINAL,
    REPORT,
    SURROGATE,
    ReportFig,
    Tuning,
    model_figs,
    report_figs,
    series_figs,
)
from neurosurrogate.report.results import SeriesView, run_names
from neurosurrogate.surrogate.bundle import SurrogateBundle

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


@dataclass(frozen=True)
class SaveEntry:
    """1 成果物 = 表示名 + 中身 (図 or 表) + 由来 (参照した評価 run と描画設定)。

    **保存名は表示名から決まる** (拡張子だけ中身の型で分かれる) ので別に持たない =
    表示と保存で名前が食い違わない。書き出し方も中身の型で決まるのでここが持つ。
    `sources`/`draw` は `meta.json` の対応する value にそのまま落ちる = 「どの
    リソースからどう描いたか」を成果物 1 件ごとに追跡できる。`draw` はどの表示設定
    (`report.figures` の `Tuning` など) でも中身を見ず `is_dataclass` でしか判定
    しない (dict 化は `save_entries` が meta.json へ書き出す境界でだけ行う) → 具体型
    への依存を持たない。
    """

    name: str
    obj: Figure | pd.DataFrame
    sources: tuple[str, ...] = ()  # 描くのに読んだ run の id (由来なしは空)
    draw: object | None = None  # 使った表示設定 dataclass (無ければ None)

    @property
    def path(self) -> str:
        """保存先ディレクトリからの相対パス。name の `/` (`<段>/<run>/<図名>` の
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


# --- 保存段の名前 (図がどの run に属するか → ディレクトリ) ----------------------


def _stage(kind_dir: str, run_id: str, names: dict[str, str], fallback: str) -> str:
    """保存段 1 つ = `<段>/<run 名>-<run id 先頭>`。

    **run id を混ぜるのは名前が同一性でないから**: MLflow の run 名は人が付け替え
    られて一意でなく (掃引違いの評価 run は同名になる)、`slug` も単射でない
    (`a b` と `a/b` は同じ)。名前だけを段にすると別 run の図が後勝ちで潰し合う。

    run が無い場合 (その場で回した結果 = 評価 run を持たない) だけ id を付けずに
    手元の名前 (系列名 / 表示ラベル) へ落として描画は通す — MLflow に無いものを
    id で名乗らない。"""
    if not run_id:
        return f"{kind_dir}/{slug(fallback)}"
    return f"{kind_dir}/{slug(names.get(run_id) or run_id)}-{run_id[:8]}"


def _dir_of(
    fig: ReportFig,
    view: SeriesView,
    names: dict[str, str],
    report_id: str,
    labels: dict[str, str],
) -> str:
    """図 1 枚 → 保存段。属する run (`kind` と `run_id`) が段を決め、id → 名前は
    渡された対応表 (`names`) からしか引かない (解けるのは mlflow を知る側だけ)。"""
    if fig.kind == MODEL:
        return _stage("models", fig.run_id, names, "")
    if fig.kind == ORIGINAL:
        return _stage("series", view.original_id, names, view.name)
    if fig.kind == SURROGATE:
        return _stage("series", view.series_id(fig.run_id), names, labels[fig.run_id])
    if fig.kind == REPORT:
        return _stage("report", report_id, names, view.name)
    # 描画側で kind が増えた/綴り違い = 黙って別の段へ落ちると気付けない
    raise ValueError(f"未知の kind {fig.kind!r} ({fig.name})")


def model_entries(
    bundles: dict[str, SurrogateBundle], tuning: Tuning, names: dict[str, str]
) -> list[SaveEntry]:
    """学習run群からモデル固有の成果物だけを組み立てる。"""
    return [
        SaveEntry(
            f"{_stage('models', fig.run_id, names, '')}/{fig.name}",
            fig.obj,
            fig.sources,
            tuning,
        )
        for fig in model_figs(bundles, tuning)
    ]


def series_entries(
    view: SeriesView,
    bundles: dict[str, SurrogateBundle],
    tuning: Tuning,
    names: dict[str, str],
) -> list[SaveEntry]:
    """評価run群から系列固有の成果物だけを組み立てる。"""
    labels = run_names(bundles)
    return [
        SaveEntry(
            f"{_dir_of(fig, view, names, '', labels)}/{fig.name}",
            fig.obj,
            fig.sources,
            tuning,
        )
        for fig in series_figs(view, bundles, tuning)
    ]


def report_entries(
    view: SeriesView,
    bundles: dict[str, SurrogateBundle],
    tuning: Tuning,
    names: dict[str, str],
    report_id: str,
) -> list[SaveEntry]:
    """レポートrunからrun横断の成果物だけを組み立てる。"""
    labels = run_names(bundles)
    return [
        SaveEntry(
            f"{_dir_of(f, view, names, report_id, labels)}/{f.name}",
            f.obj,
            f.sources,
            tuning,
        )
        for f in report_figs(view, bundles, tuning)
    ]


# --- 書き出し ------------------------------------------------------------------


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
