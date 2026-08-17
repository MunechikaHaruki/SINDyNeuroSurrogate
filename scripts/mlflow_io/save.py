"""成果物 (図 / 表) を**レポート run の artifact** として書く層 = 保存名と書き出しの
唯一の置き場所。学習 run にも評価 run にも書かない (記録した run を描画が書き換えない)。

成果物は run でなく **(レポート run, `Tuning`) に属する** — 描画時のつまみで中身が
変わるので、宛先を学習/評価 run にすると別レポートが同じ path を奪い合う。段は
`models/<run 名>/` `series/<run 名>/` とレポート自身の図 (直下) の 3 つ。
"""

from __future__ import annotations

import re
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import mlflow
from tuning import Tuning

from neurosurrogate.artifact.model import Artifacts

from . import logger

_UNSAFE = re.compile(r"[\s/\\:]+")
DRAW_FILE = "draw.json"  # そのレポート run を描いたときの表示設定 (`Tuning`)


def _slug(name: str) -> str:
    """path の 1 区切りに使う名前をパス安全へ。空名と `.` / `..` も潰す —
    **1 段は必ず 1 段**でないと「比べた 1 本 = ディレクトリ 1 つ」の対応が崩れる。"""
    out = _UNSAFE.sub("-", name.strip())
    return "-" if out in ("", ".", "..") else out


def _run_dirs(run_ids: list[str]) -> dict[str, str]:
    """学習 run_id → その run の段名 (**MLflow の run 名**)。凡例の表示名 (`meta.label`
    由来) とは別物 — 段は run を名指すのが仕事で、UI から辿れる必要がある。
    安全化後に綴りが重複するときだけ完全な run_id を足して衝突を防ぐ。"""
    slugs = {
        rid: _slug(mlflow.get_run(rid).info.run_name or rid[:8]) for rid in run_ids
    }
    duplicates = {name for name, count in Counter(slugs.values()).items() if count > 1}
    return {
        run_id: f"{name}-{run_id}" if name in duplicates else name
        for run_id, name in slugs.items()
    }


def per_run(prefix: str, artifacts: dict[str, Artifacts]) -> dict[Path, Artifacts]:
    """run 軸で開いた成果物 (学習 run_id → 図) に `<prefix>/<run 名>/` の段を付ける。
    **段名の決め方を知るのはここだけ** = `models/` も `series/` も同じ綴りで並ぶ。"""
    dirs = _run_dirs(list(artifacts))
    return {
        Path(prefix) / dirs[run_id]: run_artifacts
        for run_id, run_artifacts in artifacts.items()
    }


def save_artifacts(
    directories: dict[Path, Artifacts], report_run_id: str, tuning: Tuning
) -> list[str]:
    """成果物を全部レポート run へ書き、そのときの表示設定を `draw.json` 1 枚添える。
    返りは書いた artifact path 列 (`draw.json` も 1 件として含む = 返りがその run の
    artifact と一致する)。描き直しで同じ path は置き換わり、生成しなかった過去の path は
    残る。成果物ごとの由来は持たない (レポート run の tag が指す)。"""
    client = mlflow.MlflowClient()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        written = [
            str(file.relative_to(root))
            for path, artifacts in directories.items()
            for file in artifacts.save(root / path)
        ]
        client.log_artifacts(report_run_id, temporary)
    client.log_dict(report_run_id, asdict(tuning), DRAW_FILE)
    written.append(DRAW_FILE)
    logger.info("成果物 %d 件をレポート run へ保存: %s", len(written), report_run_id)
    return written
