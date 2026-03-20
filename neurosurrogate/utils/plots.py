# mypy: ignore-errors
import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure
from plotly.subplots import make_subplots


def draw_engine(datasets, spec, engine="matplotlib", figsize_width=10):
    """
    datasets: {"ds_name": xr.Dataset} の辞書
    spec: パネル(サブプロット)ごとの描画仕様のリスト
    """
    # --- 1. Spec に基づくデータ抽出 ---
    panels = []
    for panel_spec in spec:
        traces = []
        for tr in panel_spec["traces"]:
            # 指定されたデータセットと変数を取得
            ds = datasets[tr["ds"]]
            data = ds[tr["var"]]

            # sel が指定されていれば xarray の機能で抽出
            if "sel" in tr:
                data = data.sel(**tr["sel"])

            # プロット用の純粋な配列に変換
            x = (
                data.time.values
                if hasattr(data, "time")
                else np.arange(len(data.values.squeeze()))
            )
            y = data.values.squeeze()

            traces.append(
                {
                    "x": x,
                    "y": y,
                    "label": tr.get("label"),
                    "color": tr.get("color"),
                    "style": tr.get("style", "-"),
                }
            )
        panels.append({"ylabel": panel_spec.get("ylabel", ""), "traces": traces})

    # 最後のパネルにX軸ラベルを設定
    if panels:
        panels[-1]["xlabel"] = "Time [ms]"

    # --- 2. レンダリング ---
    if engine == "plotly":
        return _draw_plotly(panels, figsize_width)
    else:
        return _draw_matplotlib(panels, figsize_width)


def _draw_matplotlib(panels, figsize_width):
    n_rows = len(panels)
    fig = Figure(figsize=(figsize_width, 2 * n_rows))
    axs = fig.subplots(nrows=n_rows, ncols=1, sharex=True)
    if n_rows == 1:
        axs = [axs]

    for ax, panel in zip(axs, panels):
        has_legend = False
        for tr in panel["traces"]:
            ax.plot(
                tr["x"],
                tr["y"],
                label=tr["label"],
                color=tr["color"],
                linestyle=tr["style"],
            )
            if tr["label"]:
                has_legend = True

        ax.set_ylabel(panel["ylabel"])
        if has_legend:
            ax.legend()
        if panel.get("xlabel"):
            ax.set_xlabel(panel["xlabel"])

    fig.tight_layout()
    return fig


def _draw_plotly(panels, figsize_width):

    n_rows = len(panels)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[p["ylabel"] for p in panels],
    )

    for row, panel in enumerate(panels, start=1):
        for tr in panel["traces"]:
            fig.add_trace(
                go.Scatter(
                    x=tr["x"],
                    y=tr["y"],
                    name=tr["label"],
                    line=dict(
                        color=tr["color"],
                        dash=_style_to_dash(tr["style"]),
                    ),
                    showlegend=tr["label"] is not None,
                    legendgroup=str(row),
                ),
                row=row,
                col=1,
            )

    # 最後のパネルだけX軸ラベルを表示
    fig.update_xaxes(title_text="Time [ms]", row=n_rows, col=1)

    fig.update_layout(
        width=figsize_width * 96,
        height=220 * n_rows,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def _style_to_dash(style):
    # matplotlib linestyle → plotly dash
    return {
        "-": "solid",
        "--": "dash",
        ":": "dot",
        "-.": "dashdot",
    }.get(style, "solid")


def spec_simple(ds):
    spec = []
    comp_ids = np.unique(ds.coords["comp_id"].values)

    # 1. I_ext
    spec.append({"ylabel": "I_ext", "traces": [{"ds": "main", "var": "I_ext"}]})

    # 2. I_internal
    if "I_internal" in ds:
        spec.append(
            {
                "ylabel": "I_internal",
                "traces": [
                    {
                        "ds": "main",
                        "var": "I_internal",
                        "sel": {"node_id": i},
                        "label": f"Comp {i}",
                    }
                    for i in comp_ids
                ],
            }
        )

    # 3. V(t)
    spec.append(
        {
            "ylabel": "V(t) [mV]",
            "traces": [
                {
                    "ds": "main",
                    "var": "vars",
                    "sel": {"gate": False, "comp_id": i},
                    "label": f"V (Comp {i})" if len(comp_ids) > 1 else None,
                }
                for i in comp_ids
            ],
        }
    )

    # 4. Gates / Latent
    try:
        g_data = ds["vars"].sel(gate=True)
        traces = []
        for i in comp_ids:
            vars_in_comp = np.unique(g_data.sel(comp_id=i).coords["variable"].values)
            for v_name in vars_in_comp:
                traces.append(
                    {
                        "ds": "main",
                        "var": "vars",
                        "sel": {"gate": True, "comp_id": i, "variable": v_name},
                        "label": f"{v_name} (Comp {i})",
                    }
                )
        if traces:
            spec.append({"ylabel": "Gates / Latent", "traces": traces})
    except KeyError:
        pass
    return ({"main": ds}, spec)


def spec_diff(original, preprocessed, surrogate, surr_id):

    datasets = {"orig": original, "prep": preprocessed, "surr": surrogate}
    spec = []

    # 1. I_ext
    spec.append(
        {
            "ylabel": "I_ext(t)",
            "traces": [{"ds": "orig", "var": "I_ext", "color": "gold"}],
        }
    )

    # 2. V の比較
    spec.append(
        {
            "ylabel": "V [mV]",
            "traces": [
                {
                    "ds": "orig",
                    "var": "vars",
                    "sel": {"comp_id": surr_id, "variable": "V"},
                    "label": "orig V",
                    "color": "blue",
                },
                {
                    "ds": "surr",
                    "var": "vars",
                    "sel": {"comp_id": surr_id, "variable": "V"},
                    "label": "surr V",
                    "color": "red",
                    "style": "--",
                },
            ],
        }
    )

    # 3. Latent変数の比較
    prep_vars = preprocessed.coords["variable"].values.tolist()
    latent_vars = [v for v in prep_vars if v != "V"]

    for latent in latent_vars:
        spec.append(
            {
                "ylabel": latent,
                "traces": [
                    {
                        "ds": "prep",
                        "var": "vars",
                        "sel": {"variable": latent},
                        "label": f"target {latent}",
                        "color": "blue",
                    },
                    {
                        "ds": "surr",
                        "var": "vars",
                        "sel": {"comp_id": surr_id, "variable": latent},
                        "label": f"surr {latent}",
                        "color": "red",
                        "style": "--",
                    },
                ],
            }
        )

    # 4. Original の Gate 変数
    gate_names = (
        original["vars"]
        .sel(comp_id=surr_id, gate=True)
        .coords["variable"]
        .values.tolist()
    )
    spec.append(
        {
            "ylabel": "orig gates",
            "traces": [
                {
                    "ds": "orig",
                    "var": "vars",
                    "sel": {"comp_id": surr_id, "variable": name},
                    "label": name,
                }
                for name in gate_names
            ],
        }
    )
    return (datasets, spec)
