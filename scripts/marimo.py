import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    from analysis import ui as analysis

    # current_type → (amp_start, amp_stop, amp_steps)
    # 未登録 current は fallback (-5.0, 20.0, 10)
    SWEEP_DEFAULTS = {
        "lin&steady&pulse": (0, 20.0, 10),  # value [μA/cm²]
        "periodic&sinousoidal": (0, 200.0, 10),  # frequency [Hz]
    }

    TARGET_MODEL = {"hh": ["hh", "phhhp"]}

    base_ui = analysis.make_base_ui(TARGET_MODEL)
    base_ui  # noqa: B018
    return SWEEP_DEFAULTS, analysis, base_ui, mo


@app.cell
def _(SWEEP_DEFAULTS, analysis, base_ui):
    analysis.setup_mpl(base_ui["plt_style"].value)
    setting_ui = analysis.make_setting_ui(base_ui, SWEEP_DEFAULTS)
    setting_ui  # noqa: B018
    return (setting_ui,)


@app.cell
def _(analysis, base_ui):
    draw_ui = analysis.make_draw_ui(base_ui)
    draw_ui  # noqa: B018
    return (draw_ui,)


@app.cell
def _(analysis, base_ui, setting_ui):
    html_current, save_current = analysis.plot_preview(base_ui, setting_ui)
    html_current  # noqa: B018
    return (save_current,)


@app.cell
def _(save_current, save_model, save_single, save_sweep):
    # 各表示セルが「表示に使った fig」を返す → 保存も同一 object を使う
    save_items = save_model + save_current + save_single + save_sweep
    return (save_items,)


@app.cell
def _(analysis, save_items):
    save_panel = analysis.make_save_panel(save_items)
    analysis.render_save_panel(save_panel)
    return (save_panel,)


@app.cell
def _(analysis, save_items, save_panel):
    analysis.save(save_panel, save_items)
    return


@app.cell(column=1)
def _(analysis, base_ui):
    html_model, save_model = analysis.render_model_info(base_ui)
    html_model  # noqa: B018
    return (save_model,)


@app.cell
def _(analysis, base_ui, draw_ui, res_single, setting_ui):
    html_single, save_single = analysis.view_single(
        base_ui, setting_ui, res_single, draw_ui
    )
    html_single  # noqa: B018
    return (save_single,)


@app.cell
def _(analysis, base_ui, draw_ui, res_sweep, setting_ui):
    html_sweep, save_sweep = analysis.view_sweep(
        base_ui, setting_ui, res_sweep, draw_ui
    )
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
def _(analysis, base_ui, set_res_single, setting_ui):
    _new = analysis.calc_single(base_ui, setting_ui)
    if _new is not None:
        set_res_single(_new)
    return


@app.cell
def _(analysis, base_ui, set_res_sweep, setting_ui):
    _new = analysis.calc_sweep(base_ui, setting_ui)
    if _new is not None:
        set_res_sweep(_new)
    return


if __name__ == "__main__":
    app.run()
