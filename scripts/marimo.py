import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from pathlib import Path

    import marimo as mo
    from mlflow_io import (
        get_runs_df,
        load_bundles,
        load_eval_results,
        load_surrogate_model,
        run_and_log,
        sweep_siblings,
    )

    from neurosurrogate.eval import EVALS
    from neurosurrogate.metrics.report import load_and_render_report
    from neurosurrogate.runs import usable

    CONF_DIR = Path(__file__).resolve().parent / "conf"
    DRAW_JSON = CONF_DIR / "draw.json"
    STYLE_DIR = CONF_DIR / "style"
    RESULT_DIR = Path(__file__).resolve().parents[1] / "results"
    ALL_PRESETS = "(すべて)"  # preset dropdown の「絞らない」選択肢
    PLT_STYLE = "presentation"  # 描画スタイル (draw.json の関心でない = ここで固定)

    # marimo に残す操作は「run 選択」「評価」「描画」の 3 つ。評価 (→ 評価 run 保存)
    # と描画 (→ 図保存) はボタンを分け、CLI は持たない (二重管理を避け、この 2
    # ボタンが唯一の実行経路)。組み立ての中身はどれも呼び先 1 関数に畳んであり、
    # セルは呼ぶだけ。
    runs_df = get_runs_df()
    return (
        ALL_PRESETS,
        DRAW_JSON,
        EVALS,
        PLT_STYLE,
        RESULT_DIR,
        STYLE_DIR,
        load_and_render_report,
        load_bundles,
        load_eval_results,
        load_surrogate_model,
        mo,
        run_and_log,
        runs_df,
        sweep_siblings,
        usable,
    )


@app.cell
def _(ALL_PRESETS, mo, runs_df):
    # preset (yaml) 絞り込み = run_selector の上流フィルタ (一時的な選択で設定には
    # 入れない)。
    preset_ui = mo.ui.dropdown(
        options=[ALL_PRESETS, *sorted(runs_df["preset"].dropna().unique())],
        value=ALL_PRESETS,
        label="preset (yaml)",
    )
    preset_ui  # noqa: B018
    return (preset_ui,)


@app.cell
def _(ALL_PRESETS, EVALS, mo, preset, runs_df, usable):
    # marimo に残す唯一の「入力」= run を 1 件選ぶだけ。適用先 / sweep 対象 (兄弟 run)
    # は選択後に自動決定。preset で絞り、宣言された適用先 (eval entry の target) の
    # どれかへ**実際に置換できる** 代表 run (hydra sweep 親/単発 = parent_id 欠損)
    # だけ出す (子は隠す)。互換判定は `eval.usable` に委ね UI に複製しない。
    in_preset = (
        runs_df if preset == ALL_PRESETS else runs_df[runs_df["preset"] == preset]
    )
    usable_mask = in_preset["meta"].map(lambda m: usable(m, EVALS))
    reps = in_preset[usable_mask & in_preset["parent_id"].isna()]
    runs = reps[["tags.mlflow.runName", "comp_type", "run_id"]]
    run_selector = mo.ui.table(
        runs,
        label="Run (1件)",
        selection="single",
        initial_selection=[0] if len(runs) else [],
    )
    run_selector  # noqa: B018
    return (run_selector,)


@app.cell
def _(mo, sel_name):
    # 実行パネル: 評価 (→ 評価 run 保存) と描画 (→ 図保存) はボタンを分ける
    # (draw.json 調整後の再描画だけ、評価だけを別々に回せる)。どちらも CLI は持たず、
    # この 2 ボタンが唯一の実行経路 (marimo と CLI の二重管理を避ける)。保存先 (図)
    # の既定名は選択 run の runName 入り。`force` は既存の評価 run を無視して
    # 回し直す (既定はスキップ = シミュが決定的なので同じ入力は再計算しない)。
    run_panel = mo.ui.dictionary(
        {
            "dir": mo.ui.text(
                value=f"{sel_name}_result" if sel_name else "_result", label="保存先"
            ),
            "force": mo.ui.checkbox(label="評価を回し直す (force)"),
            "eval": mo.ui.run_button(label="評価 (→ 評価 run 保存)"),
            "draw": mo.ui.run_button(label="描画 (→ 図保存)"),
        }
    )
    mo.vstack(
        [
            mo.md("### 実行パネル"),
            run_panel["dir"],
            run_panel["force"],
            run_panel["eval"],
            run_panel["draw"],
        ]
    )
    return (run_panel,)


@app.cell(column=1)
def _(preset_ui):
    # **widget → plain 値の境界**。以降どの関数にも widget は渡さない。
    preset = preset_ui.value
    return (preset,)


@app.cell
def _(run_selector):
    # id は run 軸 (兄弟 run) を導く単一源、name は保存先の既定名。
    value = run_selector.value
    sel_id = value["run_id"].iloc[0] if len(value) else None
    sel_name = value["tags.mlflow.runName"].iloc[0] if len(value) else None
    return sel_id, sel_name


@app.cell
def _(sel_id, sweep_siblings):
    # 選択 run (代表) の hydra sweep 兄弟 = 自身 + 子 run_id。単発 preset は 1 件。
    # **run 軸は評価の一級の軸**なので、単発の評価でも同じ集合を重ねて比べる。
    run_ids_list = sweep_siblings(sel_id) if sel_id else []
    return (run_ids_list,)


@app.cell
def _(load_bundles, run_ids_list):
    # run 軸キー → surrogate / run_id。組み立ては mlflow_io.load_bundles 1 つに畳む。
    bundles, run_ids = load_bundles(run_ids_list)
    return bundles, run_ids


@app.cell
def _(EVALS, bundles, run_and_log, run_ids, run_panel, sel_id):
    # 評価ボタン: 評価 → 評価 run 保存だけ (描画はしない)。既に同じ入力の評価 run が
    # あればシミュごとスキップされる (force で回し直す)。
    if run_panel.value["eval"]:
        run_and_log(bundles, EVALS, run_ids, sel_id, force=run_panel.value["force"])
    return


@app.cell
def _(
    DRAW_JSON,
    PLT_STYLE,
    RESULT_DIR,
    STYLE_DIR,
    load_and_render_report,
    load_eval_results,
    load_surrogate_model,
    mo,
    run_ids_list,
    run_panel,
):
    # 描画ボタン: 評価 run + draw.json → dest へ図/表を書き出す (再シミュ無しの
    # 再描画 = draw.json 調整後もここだけ回せる)。選択した学習 run (とその sweep
    # 兄弟) が出した評価結果だけを描く。surrogate は評価 run に焼き込まれていない
    # (閉包項が要る図 diff/attractor 用に MLflow から引き直す。load_surrogate_model
    # は run_id ごとに @cache 済み)。結果読込 (mlflow 依存) だけ marimo が持ち、
    # 組立・保存は `load_and_render_report` (metrics 層) へ委譲する。
    saved = []
    if run_panel.value["draw"]:
        style_paths = [
            STYLE_DIR / "base.mplstyle",
            STYLE_DIR / f"{PLT_STYLE}.mplstyle",
        ]
        dest = RESULT_DIR / run_panel.value["dir"]
        saved = load_and_render_report(
            DRAW_JSON,
            load_eval_results(run_ids_list),
            dest,
            style_paths,
            load_surrogate_model,
        )
    (
        mo.vstack([mo.md(f"✅ `{p.relative_to(RESULT_DIR)}`") for p in saved])
        if saved
        else mo.md("(未実行)")
    )
    return


if __name__ == "__main__":
    app.run()
