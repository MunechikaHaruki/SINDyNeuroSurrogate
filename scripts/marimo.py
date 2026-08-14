import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from pathlib import Path

    import marimo as mo
    from catalog import SERIES
    from mlflow_io.report import (
        find_report_run,
        report_entries_of,
        run_and_log,
    )
    from mlflow_io.surrogate import get_runs_df, load_bundles, sweep_siblings

    from neurosurrogate.report.build import Tuning
    from neurosurrogate.report.save import save_entries
    from neurosurrogate.waveform.dynamics import METRIC_KEYS

    RESULT_DIR = Path(__file__).resolve().parents[1] / "results"
    ALL_PRESETS = "(すべて)"  # preset dropdown の「絞らない」選択肢

    # marimo に残す操作は「run 選択」「評価」「描画」の 3 つ。評価 (→ 評価 run 保存)
    # と描画 (→ 図保存) はボタンを分け、CLI は持たない (二重管理を避け、この 2
    # ボタンが唯一の実行経路)。組み立ての中身はどれも呼び先 1 関数に畳んであり、
    # セルは呼ぶだけ。
    runs_df = get_runs_df()
    return (
        ALL_PRESETS,
        METRIC_KEYS,
        RESULT_DIR,
        SERIES,
        Tuning,
        find_report_run,
        load_bundles,
        mo,
        report_entries_of,
        run_and_log,
        runs_df,
        save_entries,
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
    # 評価する系列 = **1 つだけ**。1 レポート = 1 系列 × N モデルなので、系列を選ぶ
    # ことがそのまま「どのレポートを作る / 描くか」の選択になる (複数選べると 1 回の
    # 評価が複数レポートを作り、描画の指す run_id が 1 つに定まらない)。選択肢は選んだ
    # run 群で実際に置換できる系列だけ (回せないものを選べても意味がない)。
    series_ui = mo.ui.dropdown(
        options=usable_series,
        value=usable_series[0] if usable_series else None,
        label="評価する系列 (1件)",
    )
    series_ui  # noqa: B018
    return (series_ui,)


@app.cell
def _(mo):
    # 実行パネル: 評価 (→ 評価 run 保存) と描画 (→ 図保存) はボタンを分ける
    # (宣言を書き換えた後の再描画だけ、評価だけを別々に回せる)。どちらも CLI は持たず、
    # この 2 ボタンが唯一の実行経路 (marimo と CLI の二重管理を避ける)。**widget は
    # 2 パネルに割る** — 1 つの dictionary に混ぜると評価セルが保存先の打鍵や描画
    # ボタンにも依存し、無関係な操作のたびに MLflow を引き直してしまう (reactive の
    # 依存は dictionary 単位)。`force` は既存の評価 run を無視して回し直す
    # (既定はスキップ = シミュが決定的なので同じ入力は再計算しない)。
    eval_panel = mo.ui.dictionary(
        {
            "force": mo.ui.checkbox(label="評価を回し直す (force)"),
            "eval": mo.ui.run_button(label="評価 (→ 評価 run 保存)"),
        }
    )
    # 保存先は選ばせない — 成果物の名前が `models/<学習 run>/` と
    # `report/<レポート run>/` に割れており、MLflow の run がそのまま階層になる
    # (保存名を手で付けると同じ run が別の場所に散る)。
    draw_panel = mo.ui.dictionary({"draw": mo.ui.run_button(label="描画 (→ 図保存)")})
    mo.vstack([mo.md("### 実行パネル"), *eval_panel.values(), *draw_panel.values()])
    return draw_panel, eval_panel


@app.cell
def _(METRIC_KEYS, comp_options, mo):
    # **描き方は全部ここ** (`report.Tuning` の全キー)。カタログは「何を回すか」だけを
    # 持ち、比較対象 comp も指標も図を見て決め直すもの → widget が唯一の置き場所。
    # metric の選択肢は `METRIC_KEYS` (取り出せるキーの単一源) から引くので、選んだ
    # のに生成されないキーで黙って nan の図が出ることが無い。
    tuning_ui = mo.ui.dictionary(
        {
            "eval_comp": mo.ui.dropdown(
                options=comp_options,
                value="soma" if "soma" in comp_options else None,
                label="比較対象 comp",
            ),
            # 既定は比較対象と同じ soma 1 本に絞る (空 = 全 comp は 19 区画で図が
            # 一気に重くなるので、広げるのは明示操作にする)。
            "view_comps": mo.ui.multiselect(
                options=comp_options,
                value=["soma"] if "soma" in comp_options else [],
                label="全 comp 図の表示制限 (空=全部)",
            ),
            "metric": mo.ui.dropdown(
                options=METRIC_KEYS, value="spike_count", label="点軸の折れ線の指標"
            ),
            "detail_point": mo.ui.number(0, 99, 1, value=0, label="詳細図の点 index"),
            "spike_orig": mo.ui.number(0, 99, 1, value=0, label="原系スパイク番号"),
            "spike_surr": mo.ui.number(0, 99, 1, value=0, label="置換系スパイク番号"),
            "yauto": mo.ui.checkbox(value=True, label="折れ線 y 自動"),
            "ymin": mo.ui.number(-1e4, 1e4, 1.0, value=0.0, label="y 下限"),
            "ymax": mo.ui.number(-1e4, 1e4, 1.0, value=1.0, label="y 上限"),
        }
    )
    mo.vstack([mo.md("### つまみ"), *tuning_ui.values()])
    return (tuning_ui,)


@app.cell
def _(Tuning, tuning_ui):
    # **widget → 描き方 1 値**。y レンジは「auto か否か」の 3 widget を 1 値へ畳む
    # (ドメイン側が持つのは `ylim: tuple | None` 1 つだけ = UI の都合をドメインの型に
    # 持ち込まない)。comp 未選択 (系列未選択) は空文字のまま渡し、`report_entries` の
    # 「適用先に無い comp」と同じエラー図で気付かせる。
    values = tuning_ui.value
    tuning = Tuning(
        eval_comp=values["eval_comp"] or "",
        view_comps=tuple(values["view_comps"]),
        metric=values["metric"],
        detail_point=int(values["detail_point"]),
        spike_orig=int(values["spike_orig"]),
        spike_surr=int(values["spike_surr"]),
        metric_ylim=None if values["yauto"] else (values["ymin"], values["ymax"]),
    )
    return (tuning,)


@app.cell
def _(
    RESULT_DIR,
    draw_panel,
    mo,
    report_entries_of,
    report_run_id,
    save_entries,
    tuning,
):
    # 描画ボタン: **入力はレポート run_id 1 つとつまみ (`tuning`) だけ** (再シミュ
    # 無しの再描画 = 描き方を変えてもここだけ回せる)。読込 (mlflow) と組立はすべて
    # `report_entries_of` の中で、ここに残るのは成果物の列を流すことだけ。**dest は
    # `results/` そのもの** — 階層 (`models/<学習 run>/`, `report/<レポート run>/`)
    # は成果物の名前側が持ち、MLflow の run と 1 対 1 で対応する。
    # レポート run が無い = この選択をまだ評価していない → 評価が先。
    saved = []
    if draw_panel.value["draw"] and report_run_id:
        saved = save_entries(report_entries_of(report_run_id, tuning), RESULT_DIR)
    (
        mo.vstack([mo.md(f"✅ `{p.relative_to(RESULT_DIR)}`") for p in saved])
        if saved
        else mo.md("(未実行)")
        if report_run_id
        else mo.md("この選択のレポート run が無い → 先に評価")
    )
    return


@app.cell(column=1)
def _(SERIES, series_name):
    # つまみに出す comp 名 = **選んだ系列の適用先に在る** comp だけ (適用先と噛み合わ
    # ない comp を選べない)。名前の解決は適用先を知る `SimSpec.net` に任せる。
    comp_options = sorted(SERIES[series_name].spec.net.names) if series_name else []
    return (comp_options,)


@app.cell
def _(preset_ui):
    # **widget → plain 値の境界**。以降どの関数にも widget は渡さない。
    preset = preset_ui.value
    return (preset,)


@app.cell
def _(run_selector):
    # run 軸 (兄弟 run) を導く単一源。表示名は要らない (保存段の名前は MLflow の
    # run 名を描画側が引き直す)。
    sel_id = run_selector.value["run_id"].iloc[0] if len(run_selector.value) else None
    return (sel_id,)


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
def _(series_ui):
    # widget → plain 値。**系列は 1 つ** = 回す単位も描く単位も 1 レポート。
    series_name = series_ui.value
    return (series_name,)


@app.cell
def _(SERIES, bundles):
    # 選択 run 群のどれかで**実際に置換できる**系列名 = 系列 dropdown の選択肢。
    # 置換可否の判定は `EvalSeries.replaceable` (ドメイン側) が持つ。
    usable_series = [
        name
        for name, series in SERIES.items()
        if any(series.replaceable(b.meta) for b in bundles.values())
    ]
    return (usable_series,)


@app.cell
def _(SERIES, bundles, eval_panel, find_report_run, run_and_log, series_name):
    # 評価ボタン: 選んだ 1 系列を回して波形 run + レポート run を保存 (描画はしない)。
    # 既に同じ入力の波形 run があればシミュごとスキップされる (force で回し直す)。
    # 押していないときは同じ選択 (run 群 × 系列) の既存レポートを引くので、**この
    # セルが描画の入力 = レポート run_id 1 つの単一源** (別セッションで回した結果でも
    # 選択が同じなら描ける = 描くために評価を回し直さない)。
    report_run_id = (
        (
            run_and_log(
                bundles,
                series_name,
                SERIES[series_name],
                force=eval_panel.value["force"],
            )
            if eval_panel.value["eval"]
            else find_report_run(list(bundles), SERIES[series_name])
        )
        if series_name
        else None
    )
    return (report_run_id,)


if __name__ == "__main__":
    app.run()
