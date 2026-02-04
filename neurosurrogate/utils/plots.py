# mypy: ignore-errors
import xarray as xr
from matplotlib.figure import Figure


def draw_engine(plot_configs, figsize_width=10):
    n_rows = len(plot_configs)
    fig = Figure(figsize=(figsize_width, 2 * n_rows))
    axs = fig.subplots(nrows=n_rows, ncols=1, sharex=True)
    if n_rows == 1:
        axs = [axs]

    for ax, cfg in zip(axs, plot_configs):
        # データのリスト化
        d_raw = cfg["data"]
        data_list = d_raw if isinstance(d_raw, list) else [d_raw]
        n_lines = len(data_list)

        # 凡例と色のリスト作成 (None であっても長さ n_lines のリストを保証)
        labels = cfg.get("legend")
        if labels is None:
            labels = [None] * n_lines
        elif isinstance(labels, (str, bytes)):  # 単一文字列の場合をケア
            labels = [labels]

        colors = cfg.get("colors")
        if colors is None:
            colors = [None] * n_lines

        # プロット実行
        for d, lbl, clr in zip(data_list, labels, colors):
            x = d.time if hasattr(d, "time") else range(len(d))
            ax.plot(x, d, label=lbl, color=clr)

        ax.set_ylabel(cfg["ylabel"])

        # 凡例がある（None 以外の有効な文字列がある）場合のみ表示
        if any(l is not None for l in labels):
            ax.legend()

        if cfg.get("xlabel"):
            ax.set_xlabel(cfg["xlabel"])

    fig.tight_layout()
    return fig


def plot_simple(ds):
    model_type = ds.attrs["model_type"]
    configs = []

    # 1. I_ext
    configs.append({"data": ds["I_ext"], "ylabel": "I_ext(t)"})

    # 2. I_internal (hh3)
    if model_type == "hh3" and "I_internal" in ds:
        i_int = ds["I_internal"]
        configs.append(
            {
                "data": [i_int.sel(direction=d) for d in i_int.direction.values],
                "legend": i_int.direction.values.tolist(),
                "ylabel": "I_internal",
            }
        )

    # 3. V(t)
    v_feats = ["V_pre", "V_soma", "V_post"] if model_type == "hh3" else ["V_soma"]
    configs.append(
        {
            "data": [ds["vars"].sel(variable=f) for f in v_feats],
            "legend": v_feats if len(v_feats) > 1 else None,
            "ylabel": "V(t)",
        }
    )

    # 4. Gates
    g_feats = ["latent1"] if (ds.attrs["mode"] == "surrogate") else ["M", "H", "N"]
    configs.append(
        {
            "data": [ds["vars"].sel(variable=f) for f in g_feats],
            "legend": g_feats,
            "ylabel": "gates",
            "xlabel": "Time [ms]",
        }
    )

    return draw_engine(configs)


def plot_compartment_behavior(xarray, u):
    """
    xarrayデータから特徴量ごとの時系列プロット用構成を生成し、描画する。
    """
    configs = []

    # 1. 外部入力 (I_ext)
    configs.append({"data": u, "ylabel": "I_ext(t)"})

    # 2. 各特徴量を個別の段として追加
    data_vars = xarray
    for feature_name in data_vars.get_index("features").get_level_values("variable"):
        configs.append(
            {"data": data_vars.sel(variable=feature_name), "ylabel": str(feature_name)}
        )

    # 最後の段にのみ X軸ラベルを設定
    configs[-1]["xlabel"] = "Time step"

    return draw_engine(configs)


def plot_diff(original: xr.Dataset, preprocessed: xr.DataArray, surrogate: xr.Dataset):
    configs = []
    # I_ext
    configs.append(
        {"data": original["I_ext"], "ylabel": "I_ext(t)", "colors": ["gold"]}
    )

    # 各特徴量の比較
    for feature in preprocessed.get_index("features").get_level_values("variable"):
        configs.append(
            {
                "data": [
                    preprocessed.sel(variable=feature),
                    surrogate.vars.sel(variable=feature),
                ],
                "legend": [f"Original {feature}", f"Surrogate {feature}"],
                "colors": ["blue", "red"],
                "ylabel": feature,
            }
        )

    configs[-1]["xlabel"] = "Time [ms]"
    return draw_engine(configs)
