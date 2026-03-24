import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    from scripts.flow import build_current_pipeline

    # yaml読み込み
    with open("./scripts/conf/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # UI
    selected = mo.ui.dropdown(
        options=list(cfg["current_train_pipelines"].keys()), label="experiment"
    )
    mo.hstack([selected])
    return build_current_pipeline, cfg, mo, np, plt, selected


@app.cell
def _(build_current_pipeline, cfg, mo, np, plt, selected):
    # 選択されたexpの電流を生成して表示
    pipeline = cfg["current_train_pipelines"][selected.value]

    current_cfg = {"pipeline": pipeline}

    defaults = cfg["datasets_default"]
    dt = defaults["simulator_default_dt"]
    iteration = int(defaults["simulator_default_duration"] / dt)
    current_cfg.setdefault("current_seed", defaults["default_current_seed"])
    current_cfg.setdefault("iteration", iteration)
    current_cfg.setdefault("silence_steps", int(defaults["silence_duration"] / dt))

    # パイプライン実行
    u = build_current_pipeline(current_cfg)
    t = np.arange(iteration) * dt

    fig, ax = plt.subplots()
    ax.plot(t, u)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("I_ext")
    mo.mpl.interactive(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
