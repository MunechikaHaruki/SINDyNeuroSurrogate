import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import analysis

    base_ui = analysis.make_base_ui()
    base_ui
    return analysis, base_ui


@app.cell
def _(analysis, base_ui):
    analysis.setup_mpl(base_ui["plt_style"].value)
    setting_ui = analysis.make_setting_ui(base_ui)
    analysis.render_setting_ui(setting_ui)
    return (setting_ui,)


@app.cell
def _(analysis, base_ui):
    draw_ui = analysis.make_draw_ui(base_ui)
    return (draw_ui,)


@app.cell
def _(analysis, base_ui, draw_ui, setting_ui):
    res = analysis.calc(base_ui, setting_ui, draw_ui)
    spike_ui = analysis.make_spike_ui(res, draw_ui)
    analysis.render_draw_ui(draw_ui, spike_ui)
    return res, spike_ui


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
def _(analysis, base_ui, draw_ui, res, setting_ui, spike_ui):
    html_view, save_items = analysis.view(base_ui, setting_ui, res, draw_ui, spike_ui)
    html_view
    return (save_items,)


if __name__ == "__main__":
    app.run()
