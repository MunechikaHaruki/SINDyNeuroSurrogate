import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import json
    from pathlib import Path

    import draw
    import widgets
    from mlflow_io import get_runs_df, load_bundles, sweep_siblings

    from neurosurrogate.metrics.spec import parse_evals
    from neurosurrogate.metrics.store import run_and_save

    CONF_DIR = Path(__file__).resolve().parent / "conf"
    EVAL_JSON = CONF_DIR / "eval.json"

    # marimo に残す操作は「run 選択」と「実行」の 2 つだけ。実行は評価 → artifact
    # 保存 → 描画 → 図保存を 1 クリックで済ませる (計算と描画を別ボタンに分ける理由が
    # 薄いため)。組み立ての中身はどれも呼び先 1 関数に畳んであり、セルは呼ぶだけ。
    specs = parse_evals(json.loads(EVAL_JSON.read_text()))
    runs_df = get_runs_df()
    return (
        draw,
        load_bundles,
        run_and_save,
        runs_df,
        specs,
        sweep_siblings,
        widgets,
    )


@app.cell
def _(runs_df, widgets):
    # preset (yaml) 絞り込み = run_selector の上流フィルタ (一時的な選択で設定には
    # 入れない)。
    preset_ui = widgets.make_preset_ui(runs_df)
    preset_ui  # noqa: B018
    return (preset_ui,)


@app.cell
def _(preset, runs_df, specs, widgets):
    # marimo に残す唯一の「入力」= run を 1 件選ぶだけ。適用先 / sweep 対象 (兄弟 run)
    # は選択後に自動決定。
    run_selector = widgets.make_run_ui(runs_df, preset, specs)
    run_selector  # noqa: B018
    return (run_selector,)


@app.cell
def _(sel_name, widgets):
    run_html, run_panel = widgets.make_run_panel(sel_name)
    run_html  # noqa: B018
    return (run_panel,)


@app.cell
def _(bundles, draw, run_and_save, run_ids, run_panel, specs, widgets):
    # 1 クリックで完結: 評価 → artifact 保存 → 描画 → 図保存。artifact を経由する
    # (draw.render_if は results/artifacts/ 直下すべてを読む) ので、再描画だけしたい
    # ときは marimo を経由せず CLI `uv run scripts/draw.py` を使えば足りる — marimo
    # 側で「計算だけ」「描画だけ」を別ボタンに分ける理由は薄い。
    if run_panel.value["run"]:
        run_and_save(bundles, specs, draw.ARTIFACT_DIR, run_ids)
    saved = draw.render_if(run_panel.value["run"], run_panel.value["dir"])
    widgets.written_html(saved, draw.RESULT_DIR, "(未実行)")
    return


@app.cell(column=1)
def _(preset_ui):
    # **widget → plain 値の境界**。以降どの関数にも widget は渡さない。
    preset = preset_ui.value
    return (preset,)


@app.cell
def _(run_selector, widgets):
    # id は run 軸 (兄弟 run) を導く単一源、name は保存先の既定名。取り出し方
    # (pandas indexing) は widgets 側に畳んである。
    sel_id, sel_name = widgets.selected_run(run_selector.value)
    return sel_id, sel_name


@app.cell
def _(sel_id, sweep_siblings):
    # 選択 run (代表) の hydra sweep 兄弟 = 自身 + 子 run_id。単発 preset は 1 件。
    # **run 軸は評価の一級の軸**なので、単発の評価でも同じ集合を重ねて比べる。
    run_ids_list = sweep_siblings(sel_id) if sel_id else []
    return (run_ids_list,)


@app.cell
def _(load_bundles, run_ids_list):
    # run 軸キー → surrogate / run_id。組み立ては mlflow_io.load_bundles 1 つに畳む。
    bundles, run_ids = load_bundles(run_ids_list)
    return bundles, run_ids


if __name__ == "__main__":
    app.run()
