import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from pathlib import Path

    import marimo as mo
    from analysis import actions, ui, view
    from analysis.save import panel, restore
    from mlflow_io import get_runs_df

    RESULT_DIR = Path(__file__).resolve().parent / "conf" / "surrogate" / "result"

    # current_type → (amp_start, amp_stop, amp_steps)
    # 未登録 current は fallback (-5.0, 20.0, 10)
    SWEEP_DEFAULTS = {
        "lin&steady&pulse": (0, 20.0, 10),  # value [μA/cm²]
        "periodic&sinousoidal": (0, 200.0, 10),  # frequency [Hz]
    }

    TARGET_MODEL = {"hh": ["hh", "phhhp"], "traub": ["traub19", "traub"]}

    runs_df = get_runs_df()
    return (
        RESULT_DIR,
        SWEEP_DEFAULTS,
        TARGET_MODEL,
        actions,
        mo,
        panel,
        restore,
        runs_df,
        ui,
        view,
    )


@app.cell
def _(RESULT_DIR, restore):
    # meta.json 復元パネル。dropdown 選択で即 preset をロード → 下流 UI を再構築。
    restore_html, restore_dd = restore.make_panel(RESULT_DIR)
    restore_html  # noqa: B018
    return (restore_dd,)


@app.cell
def _(restore, restore_dd):
    # 選択 meta を preset に。空選択 → None → 既定値。
    preset = restore.load(restore_dd.value)
    return (preset,)


@app.cell
def _(TARGET_MODEL, preset, runs_df, ui):
    base_ui = ui.make_base_ui(runs_df, TARGET_MODEL, preset)
    base_ui  # noqa: B018
    return (base_ui,)


@app.cell
def _(SWEEP_DEFAULTS, base_ui, preset, runs_df, ui):
    ui.setup_mpl(base_ui["plt_style"].value)
    setting_ui = ui.make_setting_ui(runs_df, base_ui, SWEEP_DEFAULTS, preset)
    setting_ui  # noqa: B018
    return (setting_ui,)


@app.cell
def _(base_ui, preset, ui):
    draw_ui = ui.make_draw_ui(base_ui, preset)
    draw_ui  # noqa: B018
    return (draw_ui,)


@app.cell
def _(base_ui, panel, setting_ui, view):
    # current preview は表示のみ (保存は mo.mpl.interactive の標準ボタンで足りる)。
    panel.render(view.plot_preview(base_ui, setting_ui))
    return


@app.cell
def _(panel, save_result):
    save_panel = panel.make_save_panel(save_result)
    panel.render_save_panel(save_panel)
    return (save_panel,)


@app.cell
def _(
    RESULT_DIR,
    base_ui,
    draw_ui,
    panel,
    restore,
    save_panel,
    save_result,
    setting_ui,
):
    panel.save(
        save_panel,
        save_result,
        RESULT_DIR,
        restore.to_meta(base_ui, setting_ui, draw_ui),
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
    view,
):
    save_result = view.view_result(
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
def _(actions, base_ui, set_res_single, setting_ui):
    _new = actions.calc_single(base_ui, setting_ui)
    if _new is not None:
        set_res_single(_new)
    return


@app.cell
def _(actions, base_ui, loaded_sweep, set_res_sweep, setting_ui):
    _new = actions.calc_sweep(base_ui, setting_ui, loaded_sweep)
    if _new is not None:
        set_res_sweep(_new)
    return


@app.cell
def _(actions, setting_ui):
    # sweep 用 run 選択の surrogate (評価サマリ + sweep で共有)。sweep 無効時は空。
    loaded_sweep = (
        actions.load_selected(setting_ui["sweep"]) if "sweep" in setting_ui else []
    )
    return (loaded_sweep,)


@app.cell
def _(actions, setting_ui):
    # single 用 run 選択の surrogate (neurograph + heatmap + 波形評価で共有)
    loaded_single = actions.load_single(setting_ui["sim"])
    return (loaded_single,)


if __name__ == "__main__":
    app.run()
