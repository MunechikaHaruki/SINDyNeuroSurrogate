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
    analysis.render_model_info(base_ui)
    return


@app.cell
def _(analysis, base_ui, setting_ui):
    analysis.plot_preview(base_ui, setting_ui)
    return


@app.cell
def _(mo):
    get_res, set_res = mo.state(None)
    return get_res, set_res


@app.cell
def _(get_res):
    res = get_res()
    return (res,)


@app.cell
def _(analysis, base_ui, set_res, setting_ui):
    _new = analysis.calc(base_ui, setting_ui)
    if _new is not None:
        set_res(_new)
    return


@app.cell
def _(analysis, base_ui, draw_ui, res, setting_ui):
    html_view, save_items = analysis.view(base_ui, setting_ui, res, draw_ui)
    html_view  # noqa: B018
    return (save_items,)


if __name__ == "__main__":
    app.run()
