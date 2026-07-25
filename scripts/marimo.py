import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from pathlib import Path

    import marimo as mo
    from analysis import ui
    from analysis.access import plt_style_of, preset_of
    from analysis.mode import single as m_single
    from analysis.mode import sweep as m_sweep
    from analysis.save import panel, restore
    from analysis.style import setup_mpl
    from mlflow_io import (
        get_runs_df,
        load_runs,
        load_surrogate_model,
        sweep_siblings,
    )

    RESULT_DIR = Path(__file__).resolve().parents[1] / "results"

    runs_df = get_runs_df()
    return (
        RESULT_DIR,
        load_runs,
        load_surrogate_model,
        m_single,
        m_sweep,
        mo,
        panel,
        plt_style_of,
        preset_of,
        restore,
        runs_df,
        setup_mpl,
        sweep_siblings,
        ui,
    )


@app.cell
def _(RESULT_DIR, restore):
    # 設定復元パネル。dropdown 選択で保存 meta.json をロード → base.json を上書き。
    restore_html, restore_dd = restore.make_panel(RESULT_DIR)
    restore_html  # noqa: B018
    return (restore_dd,)


@app.cell
def _(restore, restore_dd):
    # cfg = base.json デフォルト ← 選択 meta.json 上書き。marimo は widget を減らし、
    # シミュ入力 (current/dt/current_params/sweep 範囲) はこの cfg (ファイル) から読む。
    cfg = restore.resolve(restore_dd.value)
    return (cfg,)


@app.cell
def _(cfg, runs_df, ui):
    # preset (yaml) 絞り込み = run_selector の上流フィルタ。
    preset_ui = ui.make_preset_ui(runs_df, cfg)
    preset_ui  # noqa: B018
    return (preset_ui,)


@app.cell
def _(cfg, preset_ui, runs_df, ui):
    # marimo に残す唯一の「入力」= run を 1 件選ぶだけ。comp_type / 適用先 / sweep 対象
    # (兄弟 run) は選択後に自動決定。
    run_selector = ui.make_run_ui(runs_df, preset_ui, cfg)
    run_selector  # noqa: B018
    return (run_selector,)


@app.cell
def _(cfg, loaded_single, ui):
    # 表示調整のみ widget で残す (comp 選択肢は選択 run の代表 target から)。
    draw_ui = ui.make_draw_ui(loaded_single, cfg)
    draw_ui  # noqa: B018
    return (draw_ui,)


@app.cell
def _(draw_ui):
    # draw_ui を 1 回だけ plain dict 化。以降 domain (mode/view/access) へは dict を
    # 渡す = 描画層は marimo widget を知らない。
    draw = draw_ui.value
    return (draw,)


@app.cell
def _(mo):
    run_sim = mo.ui.run_button(label="single 実行")
    run_sim  # noqa: B018
    return (run_sim,)


@app.cell
def _(cfg, m_sweep, mo):
    # sweep 可能な電流のときだけ実行ボタン。非対応なら注記。
    run_sweep = (
        mo.ui.run_button(label="sweep 実行")
        if m_sweep.is_sweepable(cfg)
        else mo.md("(sweep 非対応電流)")
    )
    run_sweep  # noqa: B018
    return (run_sweep,)


@app.cell
def _(panel, preset_of, preset_ui, save_result):
    save_panel = panel.make_save_panel(save_result, preset_of(preset_ui))
    panel.render_save_panel(save_panel)
    return (save_panel,)


@app.cell
def _(
    RESULT_DIR,
    cfg,
    draw,
    panel,
    preset_ui,
    restore,
    run_selector,
    save_panel,
    save_result,
):
    panel.save(
        save_panel,
        save_result,
        RESULT_DIR,
        restore.to_meta(preset_ui, cfg, run_selector, draw),
    )
    return


@app.cell(column=1)
def _(
    draw,
    loaded_single,
    loaded_sweep,
    panel,
    plt_style_of,
    res_single,
    res_sweep,
    setup_mpl,
    ui,
):
    # plt_style を描画直前に適用 (draw_all より前で確実に効く)。
    setup_mpl(plt_style_of(draw))
    save_groups = ui.view_result(
        loaded_single, loaded_sweep, res_single, res_sweep, draw
    )
    # 表示は model/single/sweep をタブ分け、保存は flatten で一括対象。
    save_result = panel.flatten(save_groups)
    panel.render_groups(save_groups)
    return (save_result,)


@app.cell(column=2)
def _(cfg, draw, plt_style_of, setup_mpl, ui):
    # current preview は表示のみ。plt_style は draw (dict) から描画直前に適用。
    setup_mpl(plt_style_of(draw))
    ui.plot_preview(cfg)
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
def _(cfg, loaded_single, m_single, run_sim, set_res_single):
    if run_sim.value and loaded_single is not None:
        set_res_single(m_single.calc_eval(cfg, loaded_single))
    return


@app.cell
def _(cfg, loaded_sweep, m_sweep, run_sweep, set_res_sweep):
    if getattr(run_sweep, "value", False) and loaded_sweep:
        set_res_sweep(m_sweep.calc_sweep(cfg, loaded_sweep))
    return


@app.cell
def _(run_selector):
    # 選択 run id (single 選択テーブル → 0/1 件)。loaded_single/loaded_sweep の単一源。
    _sel = run_selector.value
    sel_id = _sel["run_id"].iloc[0] if _sel is not None and len(_sel) else None
    return (sel_id,)


@app.cell
def _(load_surrogate_model, sel_id):
    # single 用 surrogate。comp_type / 適用先 TARGET_MODEL はこの meta から自動。
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
