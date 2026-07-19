import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from pathlib import Path

    import marimo as mo
    from analysis import actions, panel, ui, view
    from mlflow_io import get_runs_df

    RESULT_DIR = (
        Path(__file__).resolve().parent.parent / "docs" / "poster" / "pic" / "result"
    )

    # current_type → (amp_start, amp_stop, amp_steps)
    # 未登録 current は fallback (-5.0, 20.0, 10)
    SWEEP_DEFAULTS = {
        "lin&steady&pulse": (0, 20.0, 10),  # value [μA/cm²]
        "periodic&sinousoidal": (0, 200.0, 10),  # frequency [Hz]
    }

    TARGET_MODEL = {"hh": ["hh", "phhhp"], "traub": ["traub19", "traub"]}

    runs_df = get_runs_df()
    base_ui = ui.make_base_ui(runs_df, TARGET_MODEL)
    base_ui  # noqa: B018
    return (
        RESULT_DIR,
        SWEEP_DEFAULTS,
        actions,
        base_ui,
        mo,
        panel,
        runs_df,
        ui,
        view,
    )


@app.cell
def _(SWEEP_DEFAULTS, base_ui, runs_df, ui):
    ui.setup_mpl(base_ui["plt_style"].value)
    setting_ui = ui.make_setting_ui(runs_df, base_ui, SWEEP_DEFAULTS)
    setting_ui  # noqa: B018
    return (setting_ui,)


@app.cell
def _(base_ui, ui):
    draw_ui = ui.make_draw_ui(base_ui)
    draw_ui  # noqa: B018
    return (draw_ui,)


@app.cell
def _(base_ui, setting_ui, view):
    html_current, save_current = view.plot_preview(base_ui, setting_ui)
    html_current  # noqa: B018
    return (save_current,)


@app.cell
def _(panel, save_current, save_single, save_sweep):
    save_groups = {
        "current": save_current,
        "single": save_single,
        "sweep": save_sweep,
    }
    save_dirs = panel.make_save_dirs(save_groups)
    save_panel = panel.make_save_panel(save_groups)
    panel.render_save_panel(save_panel, save_dirs)
    return save_dirs, save_groups, save_panel


@app.cell
def _(
    RESULT_DIR,
    base_ui,
    draw_ui,
    panel,
    save_dirs,
    save_groups,
    save_panel,
    setting_ui,
):
    panel.save(
        save_panel,
        save_groups,
        RESULT_DIR,
        save_dirs.value,
        {"base": base_ui.value, "setting": setting_ui.value, "draw": draw_ui.value},
    )
    return


@app.cell(column=1)
def _(base_ui, draw_ui, loaded_single, res_single, view):
    html_single, save_single = view.view_single(
        loaded_single, base_ui, res_single, draw_ui
    )
    html_single  # noqa: B018
    return (save_single,)


@app.cell
def _(draw_ui, loaded_sweep, res_sweep, view):
    html_sweep, save_sweep = view.view_sweep(loaded_sweep, res_sweep, draw_ui)
    html_sweep  # noqa: B018
    return (save_sweep,)


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
