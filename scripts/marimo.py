import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import json
    from pathlib import Path

    import marimo as mo
    from mlflow_io import (
        get_runs_df,
        load_bundles,
        load_surrogate_model,
        sweep_siblings,
    )

    from neurosurrogate.eval.spec import parse_evals, usable
    from neurosurrogate.eval.store import artifacts, load_all, run_and_save
    from neurosurrogate.metrics.report import ReportSpec, render_report

    CONF_DIR = Path(__file__).resolve().parent / "conf"
    EVAL_JSON = CONF_DIR / "eval.json"
    DRAW_JSON = CONF_DIR / "draw.json"
    STYLE_DIR = CONF_DIR / "style"
    RESULT_DIR = Path(__file__).resolve().parents[1] / "results"
    ARTIFACT_DIR = RESULT_DIR / "artifacts"
    ALL_PRESETS = "(すべて)"  # preset dropdown の「絞らない」選択肢
    PLT_STYLE = "presentation"  # 描画スタイル (draw.json の関心でない = ここで固定)

    # marimo に残す操作は「run 選択」「評価」「描画」の 3 つ。評価 (→ artifact 保存)
    # と描画 (→ 図保存) はボタンを分け、CLI は持たない (二重管理を避け、この 2
    # ボタンが唯一の実行経路)。組み立ての中身はどれも呼び先 1 関数に畳んであり、
    # セルは呼ぶだけ。
    specs = parse_evals(json.loads(EVAL_JSON.read_text()))
    runs_df = get_runs_df()
    return (
        ALL_PRESETS,
        ARTIFACT_DIR,
        DRAW_JSON,
        PLT_STYLE,
        RESULT_DIR,
        ReportSpec,
        STYLE_DIR,
        artifacts,
        json,
        load_all,
        load_bundles,
        load_surrogate_model,
        mo,
        render_report,
        run_and_save,
        runs_df,
        specs,
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
def _(ALL_PRESETS, mo, preset, runs_df, specs, usable):
    # marimo に残す唯一の「入力」= run を 1 件選ぶだけ。適用先 / sweep 対象 (兄弟 run)
    # は選択後に自動決定。preset で絞り、宣言された適用先 (eval entry の target) の
    # どれかへ**実際に置換できる** 代表 run (hydra sweep 親/単発 = parent_id 欠損)
    # だけ出す (子は隠す)。互換判定は `eval.spec.usable` に委ね UI に複製しない。
    in_preset = (
        runs_df if preset == ALL_PRESETS else runs_df[runs_df["preset"] == preset]
    )
    usable_mask = in_preset["meta"].map(lambda m: usable(m, specs))
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
    # 実行パネル: 評価 (→ artifact 保存) と描画 (→ 図保存) はボタンを分ける
    # (draw.json 調整後の再描画だけ、評価だけを別々に回せる)。どちらも CLI は持たず、
    # この 2 ボタンが唯一の実行経路 (marimo と CLI の二重管理を避ける)。保存先の
    # 既定名は選択 run の runName 入り。
    run_panel = mo.ui.dictionary(
        {
            "dir": mo.ui.text(
                value=f"{sel_name}_result" if sel_name else "_result", label="保存先"
            ),
            "eval": mo.ui.run_button(label="評価 (→ artifact 保存)"),
            "draw": mo.ui.run_button(label="描画 (→ 図保存)"),
        }
    )
    mo.vstack(
        [
            mo.md("### 実行パネル"),
            run_panel["dir"],
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
def _(ARTIFACT_DIR, bundles, run_and_save, run_ids, run_panel, sel_id, specs):
    # 評価ボタン: 評価 → artifact 保存だけ (描画はしない)。
    if run_panel.value["eval"]:
        run_and_save(bundles, specs, ARTIFACT_DIR, run_ids, sel_id)
    return


@app.cell
def _(
    ARTIFACT_DIR,
    DRAW_JSON,
    PLT_STYLE,
    RESULT_DIR,
    ReportSpec,
    STYLE_DIR,
    artifacts,
    json,
    load_all,
    load_surrogate_model,
    mo,
    render_report,
    run_panel,
    sel_id,
):
    # 描画ボタン: artifact + draw.json → dest へ図/表を書き出す (手元の artifact =
    # draw.json 調整後の再描画も含む)。今保存した学習 run (`sel_id`) の artifact
    # だけを描く。surrogate は artifact に焼き込まれていない (閉包項が要る図
    # diff/attractor 用に MLflow から引き直す。load_surrogate_model は run_id ごとに
    # @cache 済み)。図表の組立/保存は `render_report` (metrics 層) に委譲する。
    saved = []
    if run_panel.value["draw"]:
        report = ReportSpec.from_dict(json.loads(DRAW_JSON.read_text()))
        arts = artifacts(ARTIFACT_DIR, sel_id)
        res = load_all(arts)
        bundles_for_draw = {
            a.meta.spec.run_id: load_surrogate_model(a.meta.spec.run_id)
            for a in arts
            if a.meta.spec.run_id is not None
        }
        style_paths = [
            STYLE_DIR / "base.mplstyle",
            STYLE_DIR / f"{PLT_STYLE}.mplstyle",
        ]
        dest = RESULT_DIR / run_panel.value["dir"]
        saved = render_report(bundles_for_draw, res, report, dest, style_paths)
    (
        mo.vstack([mo.md(f"✅ `{p.relative_to(RESULT_DIR)}`") for p in saved])
        if saved
        else mo.md("(未実行)")
    )
    return


if __name__ == "__main__":
    app.run()
