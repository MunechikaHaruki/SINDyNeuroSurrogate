import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import analysis

    base_ui = analysis.make_base_ui()
    base_ui
    return analysis, base_ui, mo


@app.cell
def _(analysis, base_ui):
    analysis.setup_mpl(base_ui["plt_style"].value)
    setting_ui = analysis.make_setting_ui(base_ui)
    setting_ui
    return (setting_ui,)


@app.cell
def _(analysis, base_ui):
    draw_ui = analysis.make_draw_ui(base_ui)
    draw_ui
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
    html_view
    return (save_items,)


if __name__ == "__main__":
    app.run()
