"""成果物 (図 / 表) を**レポート run の artifact** として書く層 = 保存名と書き出しの
唯一の置き場所。

**描いたものは全部レポート run 1 本にまとまる** (1 レポート = 1 系列 × N モデル)。
比べたいのは「複数モデルの波形を並べた結果」そのものなので、束ねる単位はレポート
以外に無い。学習 run にも評価 run にも書かない = 記録した run を描画が書き換えない。

**成果物は run に属さず (レポート run, `Tuning`) に属する**: model 図も詳細図も
`view_comps` / `detail_point` といった描画時のつまみで中身が変わるため、宛先を
学習 run / 評価 run にすると、同じ run を使う別レポートが同じ path を奪い合う
(後勝ちで、どのレポートのものでもない図が残る)。レポート run 配下なら衝突しない。

run 内の path は元の 3 段をそのまま残す (`models/<run 名>/`, `series/<run 名>/`,
レポート自身の図は直下) = 1 レポートの中で「何について描いた図か」が読める。
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
    """path の 1 区切りに使う名前をパス安全へ。名前は改行や `/` を含みうるので、
    そのまま名前に混ぜると階層が割れる。

    空になる名前と `.` / `..` も潰す — **1 段は必ず 1 段**でなければ、段が消えたり
    上の階層へ抜けたりして「比べた 1 本 = ディレクトリ 1 つ」の対応が崩れる。
    """
    out = _UNSAFE.sub("-", name.strip())
    return "-" if out in ("", ".", "..") else out


def _run_dirs(run_ids: list[str]) -> dict[str, str]:
    """学習 run_id → その run の段名 (**MLflow の run 名**)。

    段名は凡例の表示名 (凡例は `meta.label` 由来) と別物にする:
    label は学習構造しか語らないので、MLflow UI で run を探すときの名前と一致せず、
    どのディレクトリがどの run のものか辿れない。段は run を名指すのが仕事。

    run 名は一意でなく、異なる名前も path 安全化後に同じ綴りになりうる。安全化した
    名前が選択内で重複するときだけ完全な run_id を足し、段の衝突を防ぐ。
    """
    names = {rid: mlflow.get_run(rid).info.run_name or rid[:8] for rid in run_ids}
    slugs = {run_id: _slug(name) for run_id, name in names.items()}
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
    """成果物を全部レポート run へ書き、そのときの表示設定を `draw.json` 1 枚に
    添える。返り値は書いた artifact path 列 (呼び出し側は表示に流すだけ)。

    描き直しで同じ path は置き換わる (`draw.json` もその 1 回分)。今回生成しなかった
    過去の path は削除しない。成果物ごとの由来は持たず、どの run から読んだかは
    レポート run の tag が既に指している。
    """
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
    logger.info("成果物 %d 件をレポート run へ保存: %s", len(written), report_run_id)
    return written
