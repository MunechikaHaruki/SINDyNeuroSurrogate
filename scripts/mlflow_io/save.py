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
from dataclasses import asdict, is_dataclass

import mlflow
import pandas as pd

from neurosurrogate.plotting import Artifact

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


def run_dirs(run_ids: list[str]) -> dict[str, str]:
    """学習 run_id → その run の段名 (**MLflow の run 名**)。

    段名は凡例の表示名 (凡例は `meta.label` 由来) と別物にする:
    label は学習構造しか語らないので、MLflow UI で run を探すときの名前と一致せず、
    どのディレクトリがどの run のものか辿れない。段は run を名指すのが仕事。

    run 名は MLflow 側で一意でないので、重複したものにだけ run_id 頭を足す
    (別 run が同じ段を奪い合わない)。
    """
    names = {rid: mlflow.get_run(rid).info.run_name or rid[:8] for rid in run_ids}
    dup = {n for n in names.values() if list(names.values()).count(n) > 1}
    return {rid: _slug(f"{n}-{rid[:8]}" if n in dup else n) for rid, n in names.items()}


def under(prefix: str, artifacts: list[Artifact]) -> list[Artifact]:
    """成果物の名前に段を付ける (`models/<run 名>/train_raw` など)。名前の `/` は
    そのまま artifact の階層になるので、包み直す型を作らずに段を表せる。"""
    return [Artifact(f"{prefix}/{a.name}", a.obj) for a in artifacts]


def _log(run_id: str, artifact: Artifact) -> str:
    """成果物 1 件をレポート run へ。**保存名は成果物の名前そのもの**で、拡張子だけ
    中身の型で分かれる = 表示と保存で名前が食い違わない。"""
    if isinstance(artifact.obj, pd.DataFrame):
        path = f"{artifact.name}.csv"
        mlflow.MlflowClient().log_text(run_id, artifact.obj.to_csv(), path)
    else:
        path = f"{artifact.name}.png"
        mlflow.MlflowClient().log_figure(run_id, artifact.obj, path)
    return path


def save_artifacts(
    artifacts: list[Artifact], report_run_id: str, tuning: object
) -> list[str]:
    """成果物を全部レポート run へ書き、そのときの表示設定を `draw.json` 1 枚に
    添える。返り値は書いた artifact path 列 (呼び出し側は表示に流すだけ)。

    描き直しは同じ path を置き換える = レポート run の artifact は**最後に描いた
    ものだけ**を持つ (`draw.json` もその 1 回分)。成果物ごとの由来は持たない —
    どの run から読んだかはレポート run の tag が既に指している。
    """
    written = [_log(report_run_id, a) for a in artifacts]
    mlflow.MlflowClient().log_dict(
        report_run_id,
        asdict(tuning) if is_dataclass(tuning) and not isinstance(tuning, type) else {},
        DRAW_FILE,
    )
    logger.info("成果物 %d 件をレポート run へ保存: %s", len(written), report_run_id)
    return written
