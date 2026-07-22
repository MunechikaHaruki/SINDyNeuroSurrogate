import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from pathlib import Path

    import marimo as mo
    from analysis import ui
    from analysis.access import (
        plt_style_of,
        preset_of,
        sim_run_selection_of,
        sweep_run_selection_of,
    )
    from analysis.mode import single as m_single
    from analysis.mode import sweep as m_sweep
    from analysis.save import panel, restore
    from analysis.style import setup_mpl
    from analysis.targets import TARGET_MODEL
    from mlflow_io import get_runs_df, load_from_selector

    RESULT_DIR = Path(__file__).resolve().parent / "conf" / "surrogate" / "result"

    runs_df = get_runs_df()
    return (
        RESULT_DIR,
        TARGET_MODEL,
        load_from_selector,
        m_single,
        m_sweep,
        mo,
        panel,
        plt_style_of,
        preset_of,
        restore,
        runs_df,
        setup_mpl,
        sim_run_selection_of,
        sweep_run_selection_of,
        ui,
    )


@app.cell
def _(RESULT_DIR, restore):
    # meta.json 復元パネル。dropdown 選択で即 preset をロード → 下流 UI を再構築。
    restore_html, restore_dd = restore.make_panel(RESULT_DIR)
    restore_html  # noqa: B018
    return (restore_dd,)


@app.cell
def _(restore, restore_dd, runs_df, ui):
    # preset (yaml) 絞り込み。base_ui より上流に置く → 選択に整合する model_pair /
    # run 一覧だけが下流で組まれる。
    # 選択 meta を preset に。空選択 → None → 既定値。
    preset = restore.load(restore_dd.value)

    preset_ui = ui.make_preset_ui(runs_df, preset)
    preset_ui  # noqa: B018
    return preset, preset_ui


@app.cell
def _(TARGET_MODEL, preset, preset_ui, runs_df, ui):
    base_ui = ui.make_base_ui(runs_df, TARGET_MODEL, preset_ui, preset)
    base_ui  # noqa: B018
    return (base_ui,)


@app.cell
def _(base_ui, plt_style_of, preset, preset_ui, runs_df, setup_mpl, ui):
    setup_mpl(plt_style_of(base_ui))
    setting_ui = ui.make_setting_ui(runs_df, base_ui, preset_ui, preset)
    setting_ui  # noqa: B018
    return (setting_ui,)


@app.cell
def _(base_ui, preset, ui):
    draw_ui = ui.make_draw_ui(base_ui, preset)
    draw_ui  # noqa: B018
    return (draw_ui,)


@app.cell
def _(base_ui, setting_ui, ui):
    # current preview は表示のみ (保存は mo.mpl.interactive の標準ボタンで足りる)。
    ui.plot_preview(base_ui, setting_ui)
    return


@app.cell
def _(panel, preset_of, preset_ui, save_result):
    save_panel = panel.make_save_panel(save_result, preset_of(preset_ui))
    panel.render_save_panel(save_panel)
    return (save_panel,)


@app.cell
def _(
    RESULT_DIR,
    base_ui,
    draw_ui,
    panel,
    preset_ui,
    restore,
    save_panel,
    save_result,
    setting_ui,
):
    panel.save(
        save_panel,
        save_result,
        RESULT_DIR,
        restore.to_meta(preset_ui, base_ui, setting_ui, draw_ui),
    )
    return


@app.cell(column=1)
def _(
    base_ui,
    draw_ui,
    loaded_single,
    loaded_sweep,
    panel,
    res_single,
    res_sweep,
    ui,
):
    save_result = ui.view_result(
        loaded_single, loaded_sweep, base_ui, res_single, res_sweep, draw_ui
    )
    panel.render(save_result)
    return (save_result,)


@app.cell(column=2)
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
def _(base_ui, m_single, set_res_single, setting_ui):
    if setting_ui["run_sim"].value:
        set_res_single(m_single.calc_eval(base_ui, setting_ui))
    return


@app.cell
def _(base_ui, loaded_sweep, m_sweep, set_res_sweep, setting_ui):
    if "run_sweep" in setting_ui and setting_ui["run_sweep"].value:
        set_res_sweep(m_sweep.calc_sweep(base_ui, setting_ui, loaded_sweep))
    return


@app.cell
def _(load_from_selector, setting_ui, sweep_run_selection_of):
    # sweep 用 run 選択の surrogate (評価サマリ + sweep で共有)。sweep 無効時は空。
    _sel = sweep_run_selection_of(setting_ui)
    loaded_sweep = load_from_selector(_sel) if _sel is not None else []
    return (loaded_sweep,)


@app.cell
def _(load_from_selector, setting_ui, sim_run_selection_of):
    # single 用 run 選択の surrogate (neurograph + heatmap + 波形評価で共有)。
    # run_selector (単一選択) の 0/1 件を 1 run or None に畳む。
    _loaded = load_from_selector(sim_run_selection_of(setting_ui))
    loaded_single = _loaded[0] if _loaded else None
    return (loaded_single,)


if __name__ == "__main__":
    app.run()
