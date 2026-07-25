"""marimo notebook の widget 層 (marimo.py と 1 対 1)。

`mo.ui.*` を作って表示するのはここだけ。`.value` を読んで plain 値へ落とすのは
marimo.py のセルなので、この層の関数は widget を**受け取らない** (返すだけ)。
計算も図の組立も持たない — シミュ設定のパースと実行は `metrics.spec`、結果図の
組立は `view.report`、保存の実体は `view.save.save_entries`。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import marimo as mo
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from mlflow_io import setup_mlflow

from neurosurrogate.metrics.spec import cfg_targets, parse_sims
from neurosurrogate.metrics.wave import DF_ROW_METRICS, SCALAR_METRICS
from neurosurrogate.neurons import MCMODELS
from neurosurrogate.surrogate.replace import replaced_names
from neurosurrogate.view.preview import current_preview_fig
from neurosurrogate.view.save import SaveEntry, save_entries

setup_mlflow()

CONF_DIR = Path(__file__).resolve().parent / "conf"
BASE_JSON = CONF_DIR / "base.json"
STYLE_DIR = CONF_DIR / "style"
STYLES = ["paper", "presentation"]
ALL_PRESETS = "(すべて)"  # preset dropdown の「絞らない」選択肢


def setup_mpl(matplotlib_style: str) -> None:
    plt.style.use(STYLE_DIR / "base.mplstyle")
    plt.style.use(STYLE_DIR / f"{matplotlib_style}.mplstyle")


def plt_style_of(cfg: dict) -> str:
    """matplotlib style 名 (描画セルが setup_mpl に渡す)。"""
    return str(cfg["draw"]["plt_style"])


def valid_or(value: object, options: object, default: object) -> object:
    """復元値が現 options に含まれれば採用、無ければ default。run 集合変化などで
    無効化した選択 (dropdown 値) を既定へ吸収する共通規約。"""
    return value if value in options else default  # type: ignore[operator]


# ---------------------------------------------------------------------------
# 設定の解決 (conf/base.json ← 保存 meta.json 上書き) と復元 widget
#
# marimo は widget を減らし「run 選択 + 実行 + 表示」に専念する。シミュ入力
# (target/current_type/dt/current_params/sweep 範囲) は widget でなく **設定 dict
# cfg** から読み、保存時は使った cfg をそのまま meta.json へ書き戻すので round-trip
# する。comp_type は cfg に持たない (選択 run の SurrogateMeta から決まる)。
# ---------------------------------------------------------------------------


def _merge(base: object, override: object) -> object:
    """deep merge: 双方 dict なら key ごとに再帰、それ以外は override 採用
    (sim/sweep のような list は entry 単位でなく丸ごと差し替え)。"""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(base.get(k), v) if k in base else v
    return out


def resolve(meta_path: str | None) -> dict:
    """base.json デフォルト ← 選択 meta.json 上書き。meta 未選択なら base のみ。"""
    base = json.loads(BASE_JSON.read_text())
    if not meta_path:
        return base
    return _merge(base, json.loads(Path(meta_path).read_text()))


def make_restore_panel(result_dir: Path) -> tuple[mo.Html, mo.ui.dropdown]:
    """復元パネル (html, dropdown)。選択肢は result_dir 直下の保存 meta.json。
    選択で即復元・空選択で既定 (base.json のみ)。run_button は click 後 False 復帰し
    gate が revert するため不採用。"""
    dropdown = mo.ui.dropdown(
        options={
            str(p.parent.relative_to(result_dir)): str(p)
            for p in sorted(result_dir.glob("*/meta.json"))
        },
        label="復元元 meta.json",
    )
    html = mo.vstack(
        [mo.md("### 設定復元 (meta.json) — 選択で base.json を上書き"), dropdown]
    )
    return html, dropdown


# ---------------------------------------------------------------------------
# Preset filter + Run 選択 (marimo に残す唯一の「入力」widget)
# ---------------------------------------------------------------------------


def make_preset_ui(runs_df: pd.DataFrame, cfg: dict) -> mo.ui.dropdown:
    """出自 preset (surrogate/*.yaml) の絞り込み dropdown。run_selector の上流フィルタ
    (preset を変えると出す run 群が変わる)。初期値は cfg (base.json⊕meta.json)。"""
    options = [ALL_PRESETS, *sorted(runs_df["preset"].dropna().unique())]
    return mo.ui.dropdown(
        options=options,
        value=valid_or(cfg.get("preset"), options, ALL_PRESETS),
        label="preset (yaml)",
    )


def preset_runs(runs_df: pd.DataFrame, preset: str) -> pd.DataFrame:
    """選択 preset の run だけに絞った runs_df (ALL_PRESETS なら素通し)。"""
    if preset == ALL_PRESETS:
        return runs_df
    return cast(pd.DataFrame, runs_df[runs_df["preset"] == preset])


def make_run_ui(runs_df: pd.DataFrame, preset: str, cfg: dict) -> mo.ui.table:
    """run 選択テーブル。preset で絞り、cfg が宣言する適用先 (sim/sweep の target)
    のどれかへ**実際に置換できる** 代表 run (sweep 親/単発 = parent_id 欠損) だけ
    出す。子は隠す。互換基準は replace ドメインのみが持ち UI に複製しない。
    初期選択=cfg run_id。"""
    in_preset = preset_runs(runs_df, preset)
    nets = [MCMODELS[t] for t in cfg_targets(cfg)]
    reps = in_preset[
        in_preset["meta"].map(lambda m: any(replaced_names(m, n) for n in nets))
        & in_preset["parent_id"].isna()
    ]
    runs = pd.DataFrame(reps[["tags.mlflow.runName", "comp_type", "run_id"]])
    wanted = set(cfg.get("run_selector") or [])
    ids = list(runs["run_id"])
    initial = [i for i, r in enumerate(ids) if r in wanted] or ([0] if ids else [])
    return mo.ui.table(
        runs, label="Run (1件)", selection="single", initial_selection=initial
    )


# ---------------------------------------------------------------------------
# Draw settings (表示調整のみ widget で残す。値は cfg["draw"] へ合流し、どの図を
# どう描くかは domain の view.report がそこから読む)
# ---------------------------------------------------------------------------


def _comp_names(cfg: dict) -> list[str]:
    """comp 選択肢 = cfg が宣言する適用先 (sim/sweep の target) の comp 名を出現順に
    重複除去した和集合。選択 run に依らず cfg だけで決まる。"""
    return list(
        dict.fromkeys(name for t in cfg_targets(cfg) for name in MCMODELS[t].names)
    )


def make_draw_ui(cfg: dict) -> mo.ui.dictionary:
    # draw_ui は 1 段フラット (ネストの益より深い添字アクセスの害が大きい)。初期値は
    # cfg["draw"] (復元 meta.json 由来)、無効化した選択は valid_or で既定へ落とす。
    names = _comp_names(cfg)
    default_comp = "soma" if "soma" in names else (names[0] if names else None)
    metrics = DF_ROW_METRICS + SCALAR_METRICS
    p = cfg.get("draw", {})
    return mo.ui.dictionary(
        {
            # plt_style は描画設定なので draw_ui に置く (描画セルが setup_mpl で適用)。
            "plt_style": mo.ui.radio(
                options=STYLES, value=p.get("plt_style", STYLES[1])
            ),
            # 既定=soma (全モデルが細胞体を "soma" と命名する共通規約)。
            "eval_comp": mo.ui.dropdown(
                options=names,
                value=valid_or(p.get("eval_comp"), names, default_comp),
                label="評価対象comp",
            ),
            # 全 comp を並べる図 (simple / train_*) の表示制限。空 = 全部 (既定)。
            # eval_comp (比較対象 1 件) とは別軸: traub19 の 19 本重ねを読める本数へ。
            "view_comps": mo.ui.multiselect(
                options=names,
                value=[c for c in p.get("view_comps", []) if c in names],
                label="表示comp (空=全部)",
            ),
            "spike_orig": mo.ui.number(
                value=int(p.get("spike_orig", 0)), step=1, label="spike orig #"
            ),
            "spike_surr": mo.ui.number(
                value=int(p.get("spike_surr", 0)), step=1, label="spike surr #"
            ),
            "sweep_metric": mo.ui.dropdown(
                options=metrics,
                value=valid_or(p.get("sweep_metric"), metrics, "spike_count"),
                label="sweep metric",
            ),
            "sweep_yauto": mo.ui.checkbox(
                value=p.get("sweep_yauto", True), label="y auto"
            ),
            "sweep_ymin": mo.ui.number(
                value=p.get("sweep_ymin", 0.0), step=1.0, label="ymin"
            ),
            "sweep_ymax": mo.ui.number(
                value=p.get("sweep_ymax", 1.0), step=1.0, label="ymax"
            ),
        }
    )


# ---------------------------------------------------------------------------
# 表示 (電流プレビュー / 結果タブ)
# ---------------------------------------------------------------------------


def plot_preview(cfg: dict) -> mo.Html:
    """sim spec ごとの電流波形プレビュー (ラベル付きで縦積み)。"""
    return mo.vstack(
        [
            mo.vstack([mo.md(f"**{label}**"), current_preview_fig(spec.dataset())])
            for label, spec in parse_sims(cfg).items()
        ]
    )


def render(entries: list[SaveEntry]) -> mo.Html:
    """save 対象をそのまま表示に流す (display と save の単一源)。"""
    blocks: list[mo.Html] = []
    for e in entries:
        body = (
            mo.mpl.interactive(e.obj)
            if isinstance(e.obj, Figure)
            else mo.ui.table(e.obj)
        )
        blocks += [mo.md(f"##### {e.name}"), body]
    return mo.vstack(blocks)


def render_groups(groups: dict[str, list[SaveEntry]]) -> mo.Html:
    """グループ (model/sim/sweep) をタブ分け表示。空グループはタブごと省く。"""
    tabs = {name: render(es) for name, es in groups.items() if es}
    return mo.ui.tabs(tabs) if tabs else mo.md("(結果なし)")


# ---------------------------------------------------------------------------
# 保存パネル (SaveEntry / 拡張子 / 書き出し / meta.json 同梱は domain の view.save。
# ここは「どれを選んだか・どこへ」の入力と表示だけ)
# ---------------------------------------------------------------------------


def make_save_panel(
    entries: list[SaveEntry], run_name: str | None
) -> tuple[mo.Html, mo.ui.dictionary]:
    """保存パネル (html, widget=保存先 + 対象複数選択 + 保存ボタン)。

    保存先の既定名は選択 run の runName 入り — run ごとに別ディレクトリへ落ち、後から
    「どの run の図か」が名前だけで分かる。
    multiselect 既定は全選択。選択を外した entry は保存対象外。
    """
    widget = mo.ui.dictionary(
        {
            "dir": mo.ui.text(
                value=f"{run_name}_result" if run_name else "_result", label="保存先"
            ),
            "select": mo.ui.multiselect(
                options=[e.name for e in entries],
                value=[e.name for e in entries],
                label="対象",
            ),
            "run": mo.ui.run_button(label="save"),
        }
    )
    html = mo.vstack(
        [mo.md("### 画像保存パネル"), widget["dir"], widget["select"], widget["run"]]
    )
    return html, widget


def save(
    opts: dict,
    entries: list[SaveEntry],
    result_dir: Path,
    meta: dict,
) -> mo.Html:
    """ボタン押下で保存パネルの値 (dir/select/run) を domain の `save_entries` へ渡し、
    書けたパスを表示。opts は marimo.py が `save_panel.value` に落としたもの。"""
    if not opts["run"]:
        return mo.md("(未保存)")
    saved = save_entries(
        entries, result_dir / opts["dir"], meta, names=set(opts["select"])
    )
    msgs = [mo.md(f"✅ `{p.relative_to(result_dir)}`") for p in saved]
    return mo.vstack(msgs) if msgs else mo.md("(未保存)")
