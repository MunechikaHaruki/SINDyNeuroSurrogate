import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    from catalog import SERIES, comp_names
    from mlflow_io.artifacts import log_report_artifacts
    from mlflow_io.report import find_report_run, run_report
    from mlflow_io.runs import ALL_PRESETS, find_selectable_runs, load_runs
    from mlflow_io.surrogate import load_bundles

    from neurosurrogate.artifact.model import Tuning
    from neurosurrogate.waveform.dynamics import METRIC_KEYS

    # 残す操作は「run 選択」「評価」「描画」の 3 つ。CLI は持たない (二重管理を避け、
    # ここが唯一の実行経路)。**セルに置くのは widget と、それを plain 値に均す 1 行
    # だけ** — 選択肢の導出は呼び先の 1 関数が持つ。
    runs_df = load_runs()
    return (
        ALL_PRESETS,
        METRIC_KEYS,
        SERIES,
        Tuning,
        comp_names,
        find_report_run,
        find_selectable_runs,
        load_bundles,
        log_report_artifacts,
        mo,
        run_report,
        runs_df,
    )


@app.cell
def _(ALL_PRESETS, SERIES, mo, runs_df):
    # **run 表に掛かる 2 つの絞り** = 同じ 1 マスクなので 1 dictionary に束ねる。
    # 系列は run に依存させない**選択の起点** (逆向きだと run の絞りが近似になる)。
    filter_ui = mo.ui.dictionary(
        {
            "series": mo.ui.dropdown(
                options=sorted(SERIES),
                value=next(iter(sorted(SERIES)), None),
                label="評価する系列 (1件)",
            ),
            "preset": mo.ui.dropdown(
                options=[ALL_PRESETS, *sorted(runs_df["preset"].dropna().unique())],
                value=ALL_PRESETS,
                label="preset (yaml)",
            ),
        }
    )
    mo.vstack([mo.md("### 絞り込み"), *filter_ui.values()])
    return (filter_ui,)


@app.cell
def _(find_selectable_runs, mo, preset, runs_df, series_name):
    # 比べたい run を N 件選ぶ (1 レポート = 1 系列 × N モデル)。何が選択肢になるかは
    # `find_selectable_runs` が持つ (絞りの条件を UI 側に複製しない)。
    runs = find_selectable_runs(runs_df, series_name, preset)
    run_selector = mo.ui.table(
        runs,
        label="Run (複数可)",
        selection="multi",
        initial_selection=[0] if len(runs) else [],
    )
    mo.vstack(
        [run_selector]
        if len(runs)
        else [run_selector, mo.md("この系列を置換できる run が無い")]
    )
    return (run_selector,)


@app.cell
def _(mo):
    # ボタンは**別 widget のまま束ねない** — 混ぜると評価セルが描画操作にも依存し
    # MLflow を引き直す。保存先は選ばせない (描いたものは全部レポート run の artifact)。
    eval_button = mo.ui.run_button(label="評価 (→ 評価 run 保存)")
    draw_button = mo.ui.run_button(label="描画 (→ 図保存)")
    mo.vstack([mo.md("### 実行パネル"), eval_button, draw_button])
    return draw_button, eval_button


@app.cell
def _(METRIC_KEYS, comp_names, mo, series_name):
    # **描き方は全部ここ** (`Tuning` の全キー)。カタログは「何を回すか」だけ。
    # metric の選択肢は `METRIC_KEYS` から引く (生成されないキーを選べない)。
    comp_options = comp_names(series_name)
    tuning_ui = mo.ui.dictionary(
        {
            "eval_comp": mo.ui.dropdown(
                options=comp_options,
                value="soma" if "soma" in comp_options else None,
                label="比較対象 comp",
            ),
            # 既定は soma 1 本 (空 = 全 comp は 19 区画で重い → 広げるのは明示操作)。
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
    # **widget → 描き方 1 値の境界**。y レンジの 3 つ (auto/下限/上限) をここで
    # `metric_ylim: tuple | None` へ畳む = UI の都合を `Tuning` に持ち込まない。
    # comp 未選択 (系列未選択) は空文字のまま渡し、描画側で設定誤りとして落とす。
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
def _(draw_button, log_report_artifacts, mo, report_run_id, tuning):
    # 描画の interface はレポート run_id + Tuning だけ (残りは呼び先が隠す)。
    saved = []
    if draw_button.value and report_run_id:
        saved = log_report_artifacts(report_run_id, tuning)
    (
        mo.vstack([mo.md(f"✅ `{p}`") for p in saved])
        if saved
        else mo.md("(未実行)")
        if report_run_id
        else mo.md("この選択のレポート run が無い → 先に評価")
    )
    return


@app.cell(column=1)
def _(filter_ui):
    # **widget → plain 値の境界**。以降どの関数にも widget は渡さない。
    series_name, preset = filter_ui.value["series"], filter_ui.value["preset"]
    return preset, series_name


@app.cell
def _(run_selector):
    # run 軸を導く単一源 = 選んだ run (与えた順)。表示名は描画側が引き直す。
    sel_ids = list(run_selector.value["run_id"])
    return (sel_ids,)


@app.cell
def _(load_bundles, sel_ids):
    # 選択 = run 軸そのもの → surrogate 群 (表示名の解決は呼び先が持つ)。
    bundles = load_bundles(sel_ids)
    return (bundles,)


@app.cell
def _(bundles, eval_button, find_report_run, run_report, series_name):
    # 評価ボタン: 1 系列を回して波形 run + レポート run を保存 (描画はしない)。押して
    # いなければ同じ選択の既存レポートを引く = **描画の入力 run_id の単一源**。
    report_run_id = (
        (
            run_report(bundles, series_name)
            if eval_button.value
            else find_report_run(series_name, tuple(bundles))
        )
        if series_name
        else None
    )
    return (report_run_id,)


if __name__ == "__main__":
    app.run()
