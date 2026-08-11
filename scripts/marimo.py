import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from pathlib import Path

    import marimo as mo
    from catalog import REPORT, SERIES
    from mlflow_io import (
        get_runs_df,
        load_bundles,
        load_report,
        load_surrogate_model,
        run_and_log,
        sweep_siblings,
    )

    from neurosurrogate.report import load_and_render_report

    RESULT_DIR = Path(__file__).resolve().parents[1] / "results"
    ALL_PRESETS = "(すべて)"  # preset dropdown の「絞らない」選択肢

    # marimo に残す操作は「run 選択」「評価」「描画」の 3 つ。評価 (→ 評価 run 保存)
    # と描画 (→ 図保存) はボタンを分け、CLI は持たない (二重管理を避け、この 2
    # ボタンが唯一の実行経路)。組み立ての中身はどれも呼び先 1 関数に畳んであり、
    # セルは呼ぶだけ。
    runs_df = get_runs_df()
    return (
        ALL_PRESETS,
        REPORT,
        RESULT_DIR,
        SERIES,
        load_and_render_report,
        load_bundles,
        load_report,
        load_surrogate_model,
        mo,
        run_and_log,
        runs_df,
        sweep_siblings,
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
def _(ALL_PRESETS, SERIES, mo, preset, runs_df):
    # marimo に残す唯一の「入力」= run を 1 件選ぶだけ。適用先 / sweep 対象 (兄弟 run)
    # は選択後に自動決定。preset で絞り、宣言された適用先 (SERIES の点の target) の
    # どれかへ**実際に置換できる** 代表 run (hydra sweep 親/単発 = parent_id 欠損)
    # だけ出す (子は隠す)。1 系列ごとの置換可否は `EvalSeries.replaceable` (ドメイン
    # 側) が持ち、「1 本でも置換できれば出す」という**選択の方針**だけがここ。
    in_preset = (
        runs_df if preset == ALL_PRESETS else runs_df[runs_df["preset"] == preset]
    )
    usable_mask = in_preset["meta"].map(
        lambda m: any(s.replaceable(m) for s in SERIES.values())
    )
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
def _(mo, usable_series):
    # 評価する系列の絞り込み = **計算入力の選択**。選択肢は選んだ run 群で実際に置換
    # できる系列だけ (回せないものを選べても意味がない)、既定は全選択なので押した
    # だけの挙動は絞る前と同じ。描画側の絞り込み (`REPORT.results`) とは役割が
    # 違う: こちらは「何を計算するか」で、あちらは「計算済みをどう描くか」。
    series_ui = mo.ui.multiselect(
        options=usable_series, value=usable_series, label="評価する系列"
    )
    series_ui  # noqa: B018
    return (series_ui,)


@app.cell
def _(mo, sel_name):
    # 実行パネル: 評価 (→ 評価 run 保存) と描画 (→ 図保存) はボタンを分ける
    # (宣言を書き換えた後の再描画だけ、評価だけを別々に回せる)。どちらも CLI は持たず、
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
    # run_id → surrogate (表示名は描画層が解く)。
    bundles = load_bundles(run_ids_list)
    return (bundles,)


@app.cell
def _(SERIES, series_ui):
    # widget → plain 値。選んだ系列名 → カタログの部分集合 (回す側が見るのはこれだけ)。
    series_names = list(series_ui.value)
    series_catalog = {name: SERIES[name] for name in series_names}
    return series_catalog, series_names


@app.cell
def _(
    REPORT,
    RESULT_DIR,
    load_and_render_report,
    load_report,
    load_surrogate_model,
    mo,
    run_ids_list,
    run_panel,
    series_names,
):
    # 描画ボタン: レポート run + 描画宣言 (`catalog.REPORT`) → dest へ図/表を
    # 書き出す (再シミュ無しの再描画 = 宣言を書き換えた後もここだけ回せる)。
    # **回した単位 = 描く単位**で、今の選択 (run 群 × 系列) のレポートを引く
    # (無ければ評価が先)。surrogate は波形 run に焼き込まれていない
    # (閉包項が要る図 diff/attractor 用に MLflow から引き直す。load_surrogate_model
    # は run_id ごとに @cache 済み)。結果読込 (mlflow 依存) だけ marimo が持ち、
    # 組立・保存は `load_and_render_report` (metrics 層) へ委譲する。
    saved = []
    if run_panel.value["draw"]:
        dest = RESULT_DIR / run_panel.value["dir"]
        saved = load_and_render_report(
            REPORT,
            load_report(run_ids_list, series_names),
            dest,
            load_surrogate_model,
        )
    (
        mo.vstack([mo.md(f"✅ `{p.relative_to(RESULT_DIR)}`") for p in saved])
        if saved
        else mo.md("(未実行)")
    )
    return


@app.cell
def _(SERIES, bundles):
    # 選択 run 群のどれかで**実際に置換できる**系列名 = 系列 multiselect の選択肢。
    # 置換可否の判定は `EvalSeries.replaceable` (ドメイン側) が持つ。
    usable_series = [
        name
        for name, series in SERIES.items()
        if any(series.replaceable(b.meta) for b in bundles.values())
    ]
    return (usable_series,)


@app.cell
def _(bundles, run_and_log, run_panel, series_catalog):
    # 評価ボタン: 評価 → 波形 run + レポート run 保存だけ (描画はしない)。既に同じ
    # 入力の波形 run があればシミュごとスキップされる (force で回し直す)。レポートは
    # 選択 (run 群 × 系列) ごとに 1 本で、同じ選択なら参照表が更新される。
    if run_panel.value["eval"]:
        run_and_log(bundles, series_catalog, force=run_panel.value["force"])
    return


if __name__ == "__main__":
    app.run()
