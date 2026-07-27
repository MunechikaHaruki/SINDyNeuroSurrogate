"""artifact (計算結果) + draw.json (描画宣言) → 図/表の書き出し。

marimo の描画ボタンが呼ぶ**唯一の描画本体** (CLI は持たない = marimo と CLI の
二重管理を避け、実行経路を 1 つに保つ)。notebook 上で図を眺めるのはやめ、描画は
「artifact を読んで PNG/CSV へ落とす」一括処理に統一する — 図は最終的に results/ の
PNG として残るものであり、marimo は MLflow run 選択・評価実行・描画の 3 操作だけに
絞る (このファイルは MLflow 非依存の domain 層ではなく scripts 側 = surrogate
ロードに MLflow を使ってよい)。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from mlflow_io import load_surrogate_model

from neurosurrogate.eval.store import Artifact, artifacts, load_all
from neurosurrogate.metrics.report import ReportSpec, eval_report, model_report
from neurosurrogate.metrics.save import save_entries
from neurosurrogate.surrogate.bundle import SurrogateBundle

CONF_DIR = Path(__file__).resolve().parent / "conf"
DRAW_JSON = CONF_DIR / "draw.json"
STYLE_DIR = CONF_DIR / "style"
RESULT_DIR = Path(__file__).resolve().parents[1] / "results"
ARTIFACT_DIR = RESULT_DIR / "artifacts"


def setup_mpl(matplotlib_style: str) -> None:
    plt.style.use(STYLE_DIR / "base.mplstyle")
    plt.style.use(STYLE_DIR / f"{matplotlib_style}.mplstyle")


def bundles_of(arts: list[Artifact]) -> dict[str, SurrogateBundle]:
    """artifact 群の出所 run を全部ロード (run_label → surrogate)。**surrogate は
    artifact に焼き込まれていない**ので、閉包項が要る図 (diff/attractor) 用に
    ここで MLflow から引き直す (`load_surrogate_model` は run_id ごとに @cache 済み
    = 同じ run を跨 label で共有しても 1 回しか DL しない)。"""
    return {a.meta.run_label: load_surrogate_model(a.meta.run_id) for a in arts}


def render(
    artifact_dir: Path = ARTIFACT_DIR,
    dest: Path = RESULT_DIR / "_result",
    draw_path: Path = DRAW_JSON,
    parent_run_id: str | None = None,
) -> list[Path]:
    """artifact + draw.json → dest へ図/表を書き出す。`parent_run_id` を渡すと
    その学習 run (親 or 孤立) 1 本分の artifact だけを描く (省略時は artifact_dir
    配下の全 run)。返り値は書いたパス列。"""
    draw_dict = json.loads(draw_path.read_text())
    report = ReportSpec.from_dict(draw_dict)
    arts = artifacts(artifact_dir, parent_run_id)
    res = load_all(arts)
    bundles = bundles_of(arts)
    setup_mpl(report.plt_style)
    entries = model_report(bundles, res, report) + eval_report(res, bundles, report)
    sources = [str(a.path.relative_to(artifact_dir)) for a in arts]
    meta = {"draw": draw_dict, "sources": sources}
    return save_entries(entries, dest, meta)


def render_if(triggered: bool, dir_name: str, parent_run_id: str) -> list[Path]:
    """marimo の保存ボタン用: 押されていなければ何もしない。dir 名の組立
    (`RESULT_DIR / dir_name`) もここに畳み、marimo 側は widget の値をそのまま渡すだけ
    にする。今保存した学習 run (`parent_run_id`) の artifact だけを描く。"""
    if not triggered:
        return []
    return render(ARTIFACT_DIR, RESULT_DIR / dir_name, parent_run_id=parent_run_id)
