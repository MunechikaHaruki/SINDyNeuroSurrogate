"""marimo notebook の widget 層 (marimo.py と 1 対 1)。

`mo.ui.*` を作って表示するのはここだけ。`.value` を読んで plain 値へ落とすのは
marimo.py のセルなので、この層の関数は widget を**受け取らない** (返すだけ)。
marimo に残す操作は「MLflow run 選択」「評価」「描画」の 3 つだけ — 表示調整や
描画の中身は `scripts/draw.py` (artifact + draw.json) に寄せ、ここは持たない。
"""

from __future__ import annotations

from pathlib import Path

import marimo as mo
import pandas as pd

from neurosurrogate.eval.spec import EvalSpec, usable

ALL_PRESETS = "(すべて)"  # preset dropdown の「絞らない」選択肢


def written_html(paths: list[Path], root: Path, empty: str) -> mo.Html:
    """書き出したパスの一覧表示 (図の保存も artifact の保存も同じ見せ方)。"""
    if not paths:
        return mo.md(empty)
    return mo.vstack([mo.md(f"✅ `{p.relative_to(root)}`") for p in paths])


def make_preset_ui(runs_df: pd.DataFrame) -> mo.ui.dropdown:
    """出自 preset (surrogate/*.yaml) の絞り込み dropdown。**run_selector を絞るだけ**
    の一時的な選択 (既定は絞らない)。"""
    return mo.ui.dropdown(
        options=[ALL_PRESETS, *sorted(runs_df["preset"].dropna().unique())],
        value=ALL_PRESETS,
        label="preset (yaml)",
    )


def make_run_ui(
    runs_df: pd.DataFrame, preset: str, specs: dict[str, EvalSpec]
) -> mo.ui.table:
    """run 選択テーブル。preset で絞り、宣言された適用先 (eval entry の target) の
    どれかへ**実際に置換できる** 代表 run (hydra sweep 親/単発 = parent_id 欠損) だけ
    出す。子は隠す。互換判定は `eval.spec.usable` (= 1 本でもシミュできるか) に
    委ね UI に複製しない。"""
    in_preset = (
        runs_df if preset == ALL_PRESETS else runs_df[runs_df["preset"] == preset]
    )
    reps = in_preset[
        in_preset["meta"].map(lambda m: usable(m, specs))
        & in_preset["parent_id"].isna()
    ]
    runs = pd.DataFrame(reps[["tags.mlflow.runName", "comp_type", "run_id"]])
    return mo.ui.table(
        runs,
        label="Run (1件)",
        selection="single",
        initial_selection=[0] if len(runs) else [],
    )


def selected_run(value: pd.DataFrame | None) -> tuple[str | None, str | None]:
    """run_selector (single 選択テーブル) の `.value` → (run_id, runName)。
    0 件選択なら (None, None)。marimo のセルは `.value` を渡すだけにし、
    中身 (pandas の取り出し) はここへ寄せる。"""
    if value is None or not len(value):
        return None, None
    return value["run_id"].iloc[0], value["tags.mlflow.runName"].iloc[0]


def make_run_panel(run_name: str | None) -> tuple[mo.Html, mo.ui.dictionary]:
    """実行パネル (html, widget=保存先 + 評価ボタン + 描画ボタン)。**評価 (→ artifact
    保存) と描画 (→ 図保存) はボタンを分ける**: 描画だけ (draw.json 調整後の再描画)
    や評価だけ (図はまだ要らない) を別々に回せる。どちらも CLI は持たず、この 2
    ボタンが唯一の実行経路 (marimo と CLI の二重管理を避ける)。保存先の既定名は
    選択 run の runName 入り。"""
    widget = mo.ui.dictionary(
        {
            "dir": mo.ui.text(
                value=f"{run_name}_result" if run_name else "_result", label="保存先"
            ),
            "eval": mo.ui.run_button(label="評価 (→ artifact 保存)"),
            "draw": mo.ui.run_button(label="描画 (→ 図保存)"),
        }
    )
    html = mo.vstack(
        [mo.md("### 実行パネル"), widget["dir"], widget["eval"], widget["draw"]]
    )
    return html, widget
