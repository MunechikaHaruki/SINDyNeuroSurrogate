"""評価結果の永続化 (artifact)。marimo/mlflow 非依存。

**計算と描画を切り離すための層**: 置換シミュ結果はこれまで marimo の state に
しか無く、カーネルを落とせば消える = 図を直すたびに再シミュしていた。ここで
結果をディスクへ落とし、描画は artifact を読むだけにする。

規約 (1 artifact = **1 surrogate run × 1 spec**、1 dir = **1 学習 run (親 or 孤立)**):

    <root>/<parent_run_id>/<label>__<run_label>__<spec_hash>/
        meta.json     # 何を回したか (入力仕様 / run_id / run_label / parent_run_id)
        data.joblib   # 点ごとの (点の値, 原系, 置換系)

dir 名は (parent_run_id, label, spec, run_label) だけで決まる = 同じ入力を回し
直せば同じ dir を上書きする (`created` は履歴表示用に meta へ残すだけで dir
命名には使わない)。

- **`parent_run_id` が dir を切る単位**: sweep なら親 run_id、単発なら自身の run_id
  (呼び出し側が渡す = marimo の run 選択 `sel_id` そのもの)。これにより
  `results/artifacts/<parent_run_id>/` の 1 dir が「回した学習 run」と 1 対 1
  対応し、`scripts/marimo.py` の描画ボタンから対象 run だけを指定して描画できる。
- **surrogate を焼き込まない**: meta に `run_id` (呼び出し側が渡す opaque な
  識別子) を書くだけ。閉包項や preprocessor が要る図 (diff/attractor) は描画側が
  run_id からロードする (その識別子が何を指すか — MLflow run 等 — は scripts 側の
  関心なのでここは知らない)。
- **run 軸は artifact の外**: run 軸 (`EvalGrid.run_labels`) は保存時に分解し、
  読込時に `load_all` が束ね直す → 後から run を 1 本足すのに全部を回し直さない。
- **保存するのは結果を作った入力** = `EvalSpec` (点ごとの dataset はそこから決定的に
  再構築できる)。単発と掃引で形が変わらない (点が 1 個か N 個かだけ)。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import xarray as xr

from ..surrogate.bundle import SurrogateBundle
from .eval import EvalGrid, EvalPoint, run_evals
from .spec import EvalSpec

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
META_FILE = "meta.json"
DATA_FILE = "data.joblib"
DTYPE = "float32"  # 波形の保存精度 (表示にも指標にも十分で容量は半分)
# float32 波形はほぼ非圧縮 (zlib で 2 割減) → 高レベルにしても時間が延びるだけ。
_COMPRESS = 1
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, kw_only=True)
class ArtifactMeta:
    """artifact の meta.json。**何を回したか**だけを持ち、波形の実体は data 側。"""

    label: str  # spec のラベル (cfg 上の名前)
    target: str  # 適用先 MC モデル名 (一覧の絞り込み用)
    run_id: str  # 出所 surrogate の識別子 (opaque。呼び出し側が MLflow run_id を渡す)
    run_label: str  # 出所 surrogate の識別名 (結果の run 軸キー)
    parent_run_id: str  # 学習 run (親 or 孤立) の識別子 = dir を切る単位
    source: dict  # 入力仕様 (EvalSpec)
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
    def spec(self) -> EvalSpec:
        """入力仕様。点ごとの dataset はここから再構築する。"""
        return EvalSpec.from_dict(self.source)

    @property
    def group_key(self) -> tuple[str, str]:
        """**同じ結果系列とみなす単位** = (label, 入力仕様)。掃引範囲を変えて回し直せば
        同じ label でも別系列 (run 軸として束ねると点の意味がずれる)。「同じ入力か」の
        正規化は `EvalSpec.key` の関心 (ここでは呼ぶだけ)。"""
        return self.label, self.spec.key()

    def dest(self, root: Path) -> Path:
        """`<parent_run_id>/<label>__<run_label>__<spec_hash>`。dir 名は
        (parent_run_id, label, spec, run_label) だけで決まる = 同じ入力を回し直せば
        必ず同じ dir を指す (`group_key` が束ねる単位と揃える)。`spec_hash` が無いと
        同じ label/run_label で spec を変えた場合に別系列を上書きしてしまう。"""
        run_dir = root / _SAFE.sub("-", self.parent_run_id)
        spec_hash = hashlib.sha1(self.spec.key().encode()).hexdigest()[:8]
        base = _SAFE.sub("-", f"{self.label}__{self.run_label}__{spec_hash}")
        return run_dir / base


@dataclass(frozen=True)
class Artifact:
    """保存済み結果への参照 (meta だけ読んだ状態)。一覧は波形を読まない。"""

    path: Path
    meta: ArtifactMeta

    @classmethod
    def read(cls, path: Path) -> Artifact:
        """`path` の meta.json だけ読む (波形は `load_data` で必要になってから)。
        スキーマ不一致は raise (呼び出し側の一覧組立が壊れた 1 件を落とす)。"""
        return cls(
            path, ArtifactMeta.from_dict(json.loads((path / META_FILE).read_text()))
        )

    def load_data(self) -> list[PointRecord]:
        """波形本体 (`data.joblib`) を読む = 点ごとの記録。
        meta だけの一覧表示では呼ばない。"""
        return joblib.load(self.path / DATA_FILE)  # type: ignore[no-any-return]


# --- 保存 ---------------------------------------------------------------------


@dataclass(frozen=True)
class PointRecord:
    """`data.joblib` に並ぶ 1 点分の記録 = (掃引点の値, 原系, 置換系)。タプルの
    位置アクセス (`d[i][2]`) で読み書きすると意味が読み取れないので名前を持つ。"""

    value: float | None
    original: xr.Dataset
    surrogate: xr.Dataset

    @classmethod
    def of(cls, point: EvalPoint, run_label: str) -> PointRecord:
        """保存前の軽量化 (波形を float32 へ = 容量半分、表示にも指標にも十分) も
        ここで済ませる。`map` はデータ変数だけを変換し座標に触らない: `astype` を
        Dataset 全体へ掛けると時間座標まで float32 になって dt が数値誤差で崩れ、
        features の MultiIndex も組み直しになる。"""

        def lighten(ds: xr.Dataset) -> xr.Dataset:
            return ds.map(lambda v: v.astype(DTYPE), keep_attrs=True)

        return cls(
            point.value, lighten(point.original), lighten(point.surrogates[run_label])
        )


def _write(root: Path, meta: ArtifactMeta, data: list[PointRecord]) -> Path:
    dest = meta.dest(root)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / META_FILE).write_text(
        json.dumps(asdict(meta), indent=2, ensure_ascii=False, default=str)
    )
    joblib.dump(data, dest / DATA_FILE, compress=_COMPRESS)
    logger.info(
        "artifact 保存: %s (%.1f MB)",
        dest.name,
        (dest / DATA_FILE).stat().st_size / 1e6,
    )
    return dest


def save(
    grid: EvalGrid,
    label: str,
    root: Path,
    run_ids: dict[str, str],
    parent_run_id: str,
) -> list[Path]:
    """1 評価結果を **run 軸ごとに 1 artifact** へ分解して保存 (run を後から足せる)。
    `run_ids` = run 軸キー → 出所 run の識別子。`parent_run_id` = 学習 run (親 or
    孤立) の識別子で、保存先 dir (`root/<parent_run_id>/...`) を切る単位。"""
    saved = []
    for run_label in grid.run_labels:
        meta = ArtifactMeta(
            label=label,
            target=grid.spec.target,
            run_id=run_ids[run_label],
            run_label=run_label,
            parent_run_id=parent_run_id,
            source=grid.spec.to_dict(),
            created=datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
        saved.append(
            _write(root, meta, [PointRecord.of(p, run_label) for p in grid.points])
        )
    return saved


def save_all(
    res: dict[str, EvalGrid], root: Path, run_ids: dict[str, str], parent_run_id: str
) -> list[Path]:
    """`run_evals` の結果をまとめて artifact 化 (label × run 軸 → artifact)。"""
    return [
        p
        for label, g in res.items()
        for p in save(g, label, root, run_ids, parent_run_id)
    ]


def run_and_save(
    bundles: dict[str, SurrogateBundle],
    specs: dict[str, EvalSpec],
    root: Path,
    run_ids: dict[str, str],
    parent_run_id: str,
) -> list[Path]:
    """評価実行 + artifact 保存を 1 呼び出しに畳む (marimo の実行ボタンが呼ぶ唯一の
    関数。**実行 = 計算 + 保存**をここで閉じ、marimo 側は結果を保持しない)。
    bundles が空 (run 未選択) なら何もしない。`parent_run_id` = marimo の run 選択
    (`sel_id`) をそのまま渡す = 1 学習 run に対して 1 artifact dir。"""
    if not bundles:
        return []
    return save_all(run_evals(bundles, specs), root, run_ids, parent_run_id)


# --- 読込 ---------------------------------------------------------------------


def artifacts(root: Path, parent_run_id: str | None = None) -> list[Artifact]:
    """`root` 配下 (`<parent_run_id>/<artifact>/`) の artifact 一覧 (meta.json だけ
    読む = 波形は触らない)。読めないもの (スキーマ変更後の残骸) は落とす = 1 件の
    失敗で一覧全体を潰さない。`parent_run_id` を渡すと学習 run 1 本分だけに絞る
    (1 dir = 1 学習 run という規約をそのままフィルタに使える)。"""
    pattern = f"{_SAFE.sub('-', parent_run_id)}/*" if parent_run_id else "*/*"
    found = []
    for d in sorted(p for p in root.glob(pattern) if (p / META_FILE).is_file()):
        try:
            found.append(Artifact.read(d))
        except Exception as e:  # noqa: BLE001 — 壊れた 1 件で一覧を落とさない
            logger.info("artifact 読込不可 %s: %s", d.name, e)
            continue
    return found


def _load_group(arts: list[Artifact]) -> EvalGrid:
    """同じ入力仕様の artifact 群 (run 軸ごと) を 1 つの `EvalGrid` へ束ね直す。
    仕様も点も揃わない artifact を混ぜると軸の意味がずれる → raise
    (混在の切り分けは `load_all` が `group_key` で済ませる。`joblib.load` と紛れない
    よう private 名にしている)。"""
    if any(a.meta.group_key != arts[0].meta.group_key for a in arts):
        raise ValueError("入力仕様の違う artifact は 1 つの結果に束ねられない")
    per_run = [(a.meta.run_label, a.load_data()) for a in arts]
    values = [r.value for r in per_run[0][1]]
    if any([r.value for r in records] != values for _, records in per_run):
        raise ValueError("点の違う artifact は 1 つの結果に束ねられない")
    return EvalGrid(
        spec=arts[0].meta.spec,
        points=[
            EvalPoint(
                rec.value,
                rec.original,
                {label: recs[i].surrogate for label, recs in per_run},
            )
            for i, rec in enumerate(per_run[0][1])
        ],
    )


def load_all(arts: list[Artifact]) -> dict[str, EvalGrid]:
    """artifact 群 → label → EvalGrid (run 軸を束ね直す)。

    束ねる単位は label でなく `group_key` (label + 入力仕様): 掃引範囲を変えて回し
    直せば同じ label でも別系列で、混ぜると点の意味がずれる。label が衝突したら
    後勝ち = 新しい系列を採る。
    """
    groups: dict[tuple[str, str], list[Artifact]] = {}
    for a in sorted(arts, key=lambda a: a.meta.created):
        groups.setdefault(a.meta.group_key, []).append(a)
    return {label: _load_group(group) for (label, _), group in groups.items()}
