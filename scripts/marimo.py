import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from pathlib import Path

    import marimo as mo
    import widgets
    from mlflow_io import (
        get_runs_df,
        load_runs,
        load_surrogate_model,
        sweep_siblings,
    )

    from neurosurrogate.metrics.spec import (
        parse_sims,
        parse_sweeps,
        run_sims,
        run_sweeps,
    )
    from neurosurrogate.view.report import result_groups
    from neurosurrogate.view.save import flatten

    RESULT_DIR = Path(__file__).resolve().parents[1] / "results"

    runs_df = get_runs_df()
    return (
        RESULT_DIR,
        flatten,
        load_runs,
        load_surrogate_model,
        mo,
        parse_sims,
        parse_sweeps,
        result_groups,
        run_sims,
        run_sweeps,
        runs_df,
        sweep_siblings,
        widgets,
    )


@app.cell
def _(RESULT_DIR, widgets):
    # 設定復元パネル。dropdown 選択で保存 meta.json をロード → base.json を上書き。
    restore_html, restore_dd = widgets.make_restore_panel(RESULT_DIR)
    restore_html  # noqa: B018
    return (restore_dd,)


@app.cell
def _(mo):
    # base.json は手で頻繁に編集する。押下で cfg セルを再実行しディスクから読み直す。
    reload_base = mo.ui.run_button(label="base.json 再読込")
    reload_base  # noqa: B018
    return (reload_base,)


@app.cell
def _(reload_base, restore_dd, widgets):
    # defaults = base.json ← 選択 meta.json 上書き (ファイル由来の解決値)。marimo は
    # widget を減らし、シミュ入力 (target/current/dt/current_params/sweep 範囲) はこの
    # ファイルから読む。widget は初期値をここから取る = cfg セルの上流。
    # reload_base 押下でこのセルが再走 → resolve がディスクを読み直す。
    reload_base  # noqa: B018
    defaults = widgets.resolve(restore_dd.value)
    return (defaults,)


@app.cell
def _(defaults, runs_df, widgets):
    # preset (yaml) 絞り込み = run_selector の上流フィルタ。
    preset_ui = widgets.make_preset_ui(runs_df, defaults)
    preset_ui  # noqa: B018
    return (preset_ui,)


@app.cell
def _(preset_ui):
    # **widget → plain 値の境界**。以降どの関数にも widget は渡さない (marimo 型を
    # 知るのは marimo.py だけ)。
    preset = preset_ui.value
    return (preset,)


@app.cell
def _(defaults, preset, runs_df, widgets):
    # marimo に残す唯一の「入力」= run を 1 件選ぶだけ。適用先 / sweep 対象 (兄弟 run)
    # は選択後に自動決定。
    run_selector = widgets.make_run_ui(runs_df, preset, defaults)
    run_selector  # noqa: B018
    return (run_selector,)


@app.cell
def _(defaults, widgets):
    # 表示調整のみ widget で残す (comp 選択肢は宣言された適用先 target から)。
    draw_ui = widgets.make_draw_ui(defaults)
    draw_ui  # noqa: B018
    return (draw_ui,)


@app.cell
def _(defaults, draw_ui):
    # cfg = ファイル由来 (sim/sweep) ⊕ widget 由来 (draw)。draw_ui を 1 回だけ plain
    # dict 化してここで合流させ、以降 domain (metrics/view) へは **cfg 1 つだけ** を
    # 渡す = 描画層は marimo widget を知らず、呼び出しも (cfg, draw) の 2 引数に
    # 割れない。出所の差を知るのはこのセルだけ。
    cfg = {**defaults, "draw": draw_ui.value}
    return (cfg,)


@app.cell
def _(mo):
    run_sim = mo.ui.run_button(label="single 実行")
    run_sim  # noqa: B018
    return (run_sim,)


@app.cell
def _(mo):
    run_sweep = mo.ui.run_button(label="sweep 実行")
    run_sweep  # noqa: B018
    return (run_sweep,)


@app.cell
def _(save_result, sel_name, widgets):
    save_html, save_panel = widgets.make_save_panel(save_result, sel_name)
    save_html  # noqa: B018
    return (save_panel,)


@app.cell
def _(RESULT_DIR, cfg, preset, save_opts, save_result, sel_ids, widgets):
    # 保存 meta.json = 描画に使った cfg (base⊕meta⊕draw) ⊕ 実際に効いた preset/run 選択
    # → base.json と同じ形で round-trip する。
    widgets.save(
        save_opts,
        save_result,
        RESULT_DIR,
        {**cfg, "preset": preset, "run_selector": sel_ids},
    )
    return


@app.cell
def _(run_selector):
    # 選択 run (single 選択テーブル → 0/1 件) を **widget から値へ**。id は
    # loaded_single/loaded_sweep の単一源、name は保存先の既定名、ids は復元用。
    _sel = run_selector.value
    _has = _sel is not None and len(_sel)
    sel_id = _sel["run_id"].iloc[0] if _has else None
    sel_name = _sel["tags.mlflow.runName"].iloc[0] if _has else None
    sel_ids = list(_sel["run_id"]) if _has else []
    return sel_id, sel_ids, sel_name


@app.cell(column=1)
def _(
    cfg,
    flatten,
    loaded_single,
    loaded_sweep,
    res_single,
    res_sweep,
    result_groups,
    widgets,
):
    # plt_style を描画直前に適用 (draw_all より前で確実に効く)。
    widgets.setup_mpl(widgets.plt_style_of(cfg))
    # 図の組立は domain (view.report)。marimo は表示 (タブ) と保存に流すだけ。
    save_groups = result_groups(cfg, loaded_single, loaded_sweep, res_single, res_sweep)
    save_result = flatten(save_groups)
    widgets.render_groups(save_groups)
    return (save_result,)


@app.cell(column=2)
def _(save_panel):
    # 保存パネルも widget → 値 (dir/select/run) にしてから渡す。
    save_opts = save_panel.value
    return (save_opts,)


@app.cell
def _(cfg, widgets):
    # current preview は表示のみ。plt_style は cfg.draw から描画直前に適用。
    widgets.setup_mpl(widgets.plt_style_of(cfg))
    widgets.plot_preview(cfg)
    return


@app.cell
def _(mo):
    # single / sweep は独立 state → 一方の実行で他方の結果表示を消さない
    get_res_single, set_res_single = mo.state(None)
    get_res_sweep, set_res_sweep = mo.state(None)
    return get_res_single, get_res_sweep, set_res_single, set_res_sweep


@app.cell
def _(get_res_single, get_res_sweep):
    res_single = get_res_single()
    res_sweep = get_res_sweep()
    return res_single, res_sweep


@app.cell
def _(cfg, loaded_single, parse_sims, run_sim, run_sims, set_res_single):
    # 設定のパースも置換シミュも domain (metrics.spec) の仕事。marimo は cfg を
    # そのまま渡して結果 (label → EvalResult) を state に置くだけ。
    if run_sim.value and loaded_single is not None:
        set_res_single(run_sims(loaded_single, parse_sims(cfg)))
    return


@app.cell
def _(cfg, loaded_sweep, parse_sweeps, run_sweep, run_sweeps, set_res_sweep):
    if run_sweep.value and loaded_sweep:
        set_res_sweep(run_sweeps(loaded_sweep, parse_sweeps(cfg)))
    return


@app.cell
def _(load_surrogate_model, sel_id):
    # single 用 surrogate。comp_type はこの meta から、適用先は cfg の sim entry から。
    loaded_single = load_surrogate_model(sel_id) if sel_id else None
    return (loaded_single,)


@app.cell
def _(sel_id, sweep_siblings):
    # 選択 run (代表) の hydra sweep 兄弟 = 自身 + 子 run_id。単発は 1 件。
    sweep_ids = sweep_siblings(sel_id) if sel_id else []
    return (sweep_ids,)


@app.cell
def _(load_runs, sweep_ids):
    # 兄弟 run をロード (sweep 実行対象)。
    loaded_sweep = load_runs(sweep_ids)
    return (loaded_sweep,)


if __name__ == "__main__":
    app.run()
