"""評価結果の永続化 (artifact)。marimo/mlflow 非依存。

**計算と描画を切り離すための層**: 置換シミュ結果はこれまで marimo の state に
しか無く、カーネルを落とせば消える = 図を直すたびに再シミュしていた。ここで
結果をディスクへ落とし、描画は artifact を読むだけにする。

規約 (1 artifact = **1 SimSpec = 1 joblib ファイル**、
1 dir = **1 学習 run (親 or 孤立)**):

    <root>/<parent_run_id>/<label>__<run_label|original>__<spec_hash>.joblib
        {"meta": ArtifactMeta の dict, "dataset": xr.Dataset} を 1 ファイルに保存

meta (何を回したか) と波形を別ファイルに分けない。ファイル名は (parent_run_id,
label, spec, run_label) だけで決まる = 同じ入力を回し直せば同じファイルを
上書きする (`created` は履歴表示用に meta へ残すだけでファイル名には使わない)。

- **`parent_run_id` が dir を切る単位**: sweep なら親 run_id、単発なら自身の run_id
  (呼び出し側が渡す = marimo の run 選択 `sel_id` そのもの)。これにより
  `results/artifacts/<parent_run_id>/` の 1 dir が「回した学習 run」と 1 対 1
  対応し、`scripts/marimo.py` の描画ボタンから対象 run だけを指定して描画できる。
- **surrogate を焼き込まない**: `SimSpec.run_id` (MLflow run_id そのもの) を持つだけ。
  閉包項や preprocessor が要る図 (diff/attractor) は描画側が run_id からロードする
  (その識別子が何を指すか — MLflow run 等 — は scripts 側の関心なのでここは知らない)。
- **保存するのは結果を作った入力** = `SimSpec` (dataset はそこから決定的に
  再構築できる)。掃引点も run もすでに `SimSpec` に確定済みなので、単発/掃引/
  原系/置換系のどれでも同じ形。
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import xarray as xr

from ..surrogate.bundle import SurrogateBundle
from .run import SimKey, expand, simulate
from .spec import SimSpec

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3
DTYPE = "float32"  # 波形の保存精度 (表示にも指標にも十分で容量は半分)
# float32 波形はほぼ非圧縮 (zlib で 2 割減) → 高レベルにしても時間が延びるだけ。
_COMPRESS = 1
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_ORIGINAL = "original"  # 原系 (run_label=None) のファイル名に使う固定文字列


@dataclass(frozen=True, kw_only=True)
class ArtifactMeta:
    """artifact の meta.json。**何を回したか**だけを持ち、波形の実体は data 側。"""

    label: str  # spec のラベル (掃引展開後の eval.json キー)
    run_label: str | None  # 出所 surrogate の表示名 (結果の run 軸キー、原系は None)
    parent_run_id: str  # 学習 run (親 or 孤立) の識別子 = dir を切る単位
    source: dict  # 入力仕様 (SimSpec)
    created: str
    dtype: str = DTYPE
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, d: dict) -> ArtifactMeta:
        """スキーマが変わった artifact は読まずに落とす (黙って別物を描かない)。"""
        if int(d.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(f"schema_version 不一致: {d.get('schema_version')}")
        return cls(**d)

    @property
    def spec(self) -> SimSpec:
        """入力仕様。dataset はここから再構築する。"""
        return SimSpec.from_dict(self.source)

    def dest(self, root: Path) -> Path:
        """`<parent_run_id>/<label>__<run_label|original>__<spec_hash>.joblib`。
        ファイル名は (parent_run_id, label, spec, run_label) だけで決まる = 同じ
        入力を回し直せば必ず同じファイルを指す。`spec_hash` が無いと同じ
        label/run_label で spec を変えた場合に別系列を上書きしてしまう。"""
        run_dir = root / _SAFE.sub("-", self.parent_run_id)
        spec_hash = hashlib.sha1(self.spec.key().encode()).hexdigest()[:8]
        run = self.run_label or _ORIGINAL
        base = _SAFE.sub("-", f"{self.label}__{run}__{spec_hash}")
        return run_dir / f"{base}.joblib"


@dataclass(frozen=True)
class Artifact:
    """保存済み結果への参照 (meta だけ読んだ状態)。一覧は波形を読まない。"""

    path: Path
    meta: ArtifactMeta

    @classmethod
    def read(cls, path: Path) -> Artifact:
        """`path` の meta を読む (1 ファイルに波形も同居するため実体は丸ごと読む
        が、返すのは meta のみ)。スキーマ不一致は raise (呼び出し側の一覧組立が
        壊れた 1 件を落とす)。"""
        meta = ArtifactMeta.from_dict(joblib.load(path)["meta"])
        return cls(path, meta)

    def load_data(self) -> xr.Dataset:
        """波形本体を読む。meta だけの一覧表示では呼ばない。"""
        return joblib.load(self.path)["dataset"]  # type: ignore[no-any-return]


@dataclass(frozen=True)
class SimResult:
    """1 SimSpec 分の実行結果 = 仕様 + 表示名 + 波形。"""

    spec: SimSpec
    run_label: str | None  # 表示名 (凡例/行見出し)。None=原系
    dataset: xr.Dataset
    source: Path | None = None  # 読込元 artifact パス (実行直後は無い = None)


# --- 保存 ---------------------------------------------------------------------


def _write(root: Path, meta: ArtifactMeta, ds: xr.Dataset) -> Path:
    dest = meta.dest(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lightened = ds.map(lambda v: v.astype(DTYPE), keep_attrs=True)
    joblib.dump({"meta": asdict(meta), "dataset": lightened}, dest, compress=_COMPRESS)
    logger.info("artifact 保存: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def save(label: str, result: SimResult, root: Path, parent_run_id: str) -> Path:
    """1 SimResult を 1 artifact へ保存する。`label` は eval.json のキー (掃引展開後
    の `name#i` 込み) — `result.spec.name` は掃引しても不変な系列名なので label
    とは別物 (label が dir 名 = artifact の同一系列判定に使う単位)。"""
    meta = ArtifactMeta(
        label=label,
        run_label=result.run_label,
        parent_run_id=parent_run_id,
        source=result.spec.to_dict(),
        created=datetime.now().strftime("%Y%m%d-%H%M%S.%f"),
    )
    return _write(root, meta, result.dataset)


def save_all(
    results: dict[SimKey, SimResult], root: Path, parent_run_id: str
) -> list[Path]:
    """`SimKey → SimResult` をまとめて artifact 化 (label × run 軸 → artifact)。"""
    return [
        save(label, result, root, parent_run_id)
        for (label, _run_id), result in results.items()
    ]


def run_and_save(
    bundles: dict[str, SurrogateBundle],
    specs: dict[str, SimSpec],
    root: Path,
    run_labels: dict[str, str],
    parent_run_id: str,
) -> list[Path]:
    """評価実行 + artifact 保存を 1 呼び出しに畳む (marimo の実行ボタンが呼ぶ唯一の
    関数。**実行 = 計算 + 保存**をここで閉じ、marimo 側は結果を保持しない)。
    bundles が空 (run 未選択) なら何もしない。`run_labels` = run_id → 表示名。
    `parent_run_id` = marimo の run 選択 (`sel_id`) をそのまま渡す = 1 学習 run
    に対して 1 artifact dir。"""
    if not bundles:
        return []
    expanded = expand(specs, bundles)
    results = {
        key: SimResult(
            spec,
            run_labels[key[1]] if key[1] is not None else None,
            simulate(spec, bundles[key[1]] if key[1] is not None else None),
        )
        for key, spec in expanded.items()
    }
    return save_all(results, root, parent_run_id)


# --- 読込 ---------------------------------------------------------------------


def artifacts(root: Path, parent_run_id: str | None = None) -> list[Artifact]:
    """`root` 配下 (`<parent_run_id>/<artifact>.joblib`) の artifact 一覧。読めない
    もの (スキーマ変更後の残骸) は落とす = 1 件の失敗で一覧全体を潰さない。
    `parent_run_id` を渡すと学習 run 1 本分だけに絞る (1 dir = 1 学習 run という
    規約をそのままフィルタに使える)。"""
    pattern = (
        f"{_SAFE.sub('-', parent_run_id)}/*.joblib" if parent_run_id else "*/*.joblib"
    )
    found = []
    for p in sorted(root.glob(pattern)):
        try:
            found.append(Artifact.read(p))
        except Exception as e:  # noqa: BLE001 — 壊れた 1 件で一覧を落とさない
            logger.info("artifact 読込不可 %s: %s", p.name, e)
            continue
    return found


def load_all(arts: list[Artifact]) -> dict[SimKey, SimResult]:
    """artifact 群 → `SimKey → SimResult`。label 衝突 (同じ label/run_label で入力
    仕様が違う artifact が混在) は新しいもの (`created` が新しい方) が勝つ。"""
    out: dict[SimKey, SimResult] = {}
    for a in sorted(arts, key=lambda a: a.meta.created):
        spec = a.meta.spec
        key = (a.meta.label, spec.run_id)
        out[key] = SimResult(spec, a.meta.run_label, a.load_data(), source=a.path)
    return out
