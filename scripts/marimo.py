import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    from catalog import SERIES, comp_names
    from mlflow_io.report import log_report_artifacts, run_report
    from mlflow_io.runs import ALL_PRESETS, find_selectable_runs, load_runs

    from neurosurrogate.waveform.dynamics import METRIC_KEYS

    # 残す操作は「run 選択」「レポート」の 2 つ。CLI は持たない (二重管理を避け、
    # ここが唯一の実行経路)。**セルに置くのは widget と、それを plain 値に均す 1 行
    # だけ** — 選択肢の導出は呼び先の 1 関数が持つ。
    runs_df = load_runs()
    return (
        ALL_PRESETS,
        METRIC_KEYS,
        SERIES,
        comp_names,
        find_selectable_runs,
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
def _(
    METRIC_KEYS,
    comp_names,
    find_selectable_runs,
    mo,
    preset,
    runs_df,
    series_name,
):
    # **1 レポートの入力は全部ここ**: 比べる run を N 件選び (1 レポート = 1 系列 ×
    # N モデル)、その描き方 (つまみの全キー) を隣で決める。どちらも選んだ 1 系列で
    # 選択肢が決まるので、離すと同じ series_name を 2 セルが読むだけになる。
    # 何が選択肢になるかは `find_selectable_runs` が持ち (絞りの条件を UI 側に
    # 複製しない)、metric は `METRIC_KEYS` から引く (生成されないキーを選べない)。
    runs = find_selectable_runs(runs_df, series_name, preset)
    run_selector = mo.ui.table(
        runs,
        label="Run (複数可)",
        selection="multi",
        initial_selection=[0] if len(runs) else [],
    )
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
    mo.vstack(
        [
            run_selector,
            *([] if len(runs) else [mo.md("この系列を置換できる run が無い")]),
            mo.md("### つまみ"),
            *tuning_ui.values(),
        ]
    )
    return run_selector, tuning_ui


@app.cell
def _(mo):
    # 操作は 1 つ = **レポートを 1 本作る**。評価と描画は同じ 1 回の意図の前半と後半で、
    # 分けても片方だけ押す使い方をしない (分けると run_id を跨いで持ち回る必要も出る)。
    # 保存先は選ばせない (描いたものは全部レポート run の artifact)。
    report_button = mo.ui.run_button(label="レポート (評価 → 図保存)")
    mo.vstack([mo.md("### 実行パネル"), report_button])
    return (report_button,)


@app.cell
def _(
    log_report_artifacts,
    mo,
    report_button,
    run_report,
    sel_ids,
    series_name,
    tuning_ui,
):
    # **レポート 1 本 = 評価 → 描画**。前半は重い (シミュ + 波形 run 保存)、後半は
    # レポート run へ図を書くだけなので、どちらを走っているかだけ spinner に出す
    # (進捗率は点数にも図の枚数にも依らず出せないので出さない)。
    saved = []
    if report_button.value and series_name:
        with mo.status.spinner(title="評価中 (シミュ + 波形 run 保存)") as progress:
            report_run_id = run_report(tuple(sel_ids), series_name)
            progress.update(title="描画中 (図をレポート run へ保存)")
            saved = log_report_artifacts(report_run_id, tuning_ui.value)
    mo.vstack([mo.md(f"✅ `{p}`") for p in saved]) if saved else mo.md("(未実行)")
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


if __name__ == "__main__":
    app.run()
