"""レポート run の**成果物 (図/表) の書き出し** = marimo の描画ボタンの呼び先。

experiment も同一性も持たない (それは `report` の関心) — ここが知るのは
**段名を何と綴るか** (学習 run の run 名を引く) と、**どこへ書くか** の 2 つだけ。
どの図をどの段に置くかと保存規約はドメイン側 (`artifact.bundle.build_report`)。

宛先はレポート run に固定する: 成果物は run でなく (レポート run, `Tuning`) に属する
ので、学習/波形 run に書くと別レポートが同じ path を奪い合い、記録した run を描画が
書き換えることにもなる。
"""

from __future__ import annotations

import re
import tempfile
from collections import Counter
from pathlib import Path

import mlflow

from neurosurrogate.artifact.bundle import build_report
from neurosurrogate.artifact.model import Tuning

from . import logger
from .report import load_report
from .surrogate import load_bundles

_UNSAFE = re.compile(r"[\s/\\:]+")


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


def log_report_artifacts(report_run_id: str, tuning: Tuning) -> list[str]:
    """レポート run を描画し、全成果物を同じ run へ書き足す唯一の interface。呼ぶ側が
    知るのは id と描画条件だけで、参照解決も surrogate ロードも段構成もここから先。

    そのときの表示設定を `draw.json` 1 枚添える。返りは書いた artifact path 列
    (`draw.json` も 1 件 = 返りがその run の artifact と一致する)。描き直しで同じ path
    は置き換わり、生成しなかった過去の path は残る。成果物ごとの由来は持たない
    (レポート run の tag が指す)。"""
    view = load_report(report_run_id)
    bundles = load_bundles(view.run_ids)
    with tempfile.TemporaryDirectory() as temporary:
        written = [
            str(file)
            for file in build_report(
                view, bundles, tuning, _run_dirs(view.run_ids)
            ).save(Path(temporary))
        ]
        mlflow.MlflowClient().log_artifacts(report_run_id, temporary)
    logger.info("成果物 %d 件をレポート run へ保存: %s", len(written), report_run_id)
    return written
