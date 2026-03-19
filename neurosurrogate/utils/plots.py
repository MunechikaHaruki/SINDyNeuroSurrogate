# mypy: ignore-errors
import numpy as np
import xarray as xr
from matplotlib.figure import Figure


def draw_engine(plot_configs, figsize_width=10):
    # 時間軸のラベルを指定
    plot_configs[-1]["xlabel"] = "Time [ms]"

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
        if any(label is not None for label in labels):
            ax.legend()

        if cfg.get("xlabel"):
            ax.set_xlabel(cfg["xlabel"])

    fig.tight_layout()
    return fig


def plot_simple(ds):
    configs = []

    # コンパートメントIDのリストを動的に取得 (例: [0, 1, 2])
    # ※もしI_internalの次元が "node_id" のままなら、ここを "node_id" に読み替えてください
    comp_ids = np.unique(ds.coords["comp_id"].values)

    # 1. I_ext (外部電流)
    configs.append({"data": ds["I_ext"], "ylabel": "I_ext"})

    # 2. I_internal (内部電流 / トータル電流)
    if "I_internal" in ds:
        configs.append(
            {
                "data": [ds["I_internal"].sel(node_id=i) for i in comp_ids],
                "legend": [f"Comp {i}" for i in comp_ids],
                "ylabel": "I_internal",
            }
        )

    # 3. V(t) (膜電位)
    # gate=False のデータを抽出し、各コンパートメントごとにリスト化
    v_data = ds["vars"].sel(gate=False)
    configs.append(
        {
            "data": [v_data.sel(comp_id=i) for i in comp_ids],
            "legend": [f"V (Comp {i})" for i in comp_ids]
            if len(comp_ids) > 1
            else None,
            "ylabel": "V(t) [mV]",
        }
    )

    # 4. Gates / Latent (ゲート変数・潜在変数)
    # gate=True のデータが存在する場合のみプロットを作成する
    try:
        g_data = ds["vars"].sel(gate=True)
        g_plot_data = []
        g_legends = []

        # コンパートメントごとに、存在するゲート変数をすべて取り出す
        for i in comp_ids:
            # そのコンパートメントに存在する変数名のリストを取得 (例: ["M", "H", "N"] や ["latent1"])
            vars_in_comp = np.unique(g_data.sel(comp_id=i).coords["variable"].values)

            for v_name in vars_in_comp:
                g_plot_data.append(g_data.sel(comp_id=i, variable=v_name))
                g_legends.append(f"{v_name} (Comp {i})")

        configs.append(
            {
                "data": g_plot_data,
                "legend": g_legends,
                "ylabel": "Gates / Latent",
            }
        )
    except KeyError:
        # パッシブモデルなど、ゲート変数が一つもない場合はスキップ
        pass

    return draw_engine(configs)


def plot_diff(
    original: xr.Dataset,
    preprocessed: xr.DataArray,
    surrogate: xr.Dataset,
    surr_id=None,
):
    configs = []
    # I_ext
    configs.append(
        {"data": original["I_ext"], "ylabel": "I_ext(t)", "colors": ["gold"]}
    )
    if surr_id is None:
        surr_id = surrogate.attrs["surr_ids"][0]
    # originalとsurrogateのVの時間変化
    orig_v = original["vars"].sel(comp_id=surr_id, variable="V").squeeze()
    surr_v = surrogate["vars"].sel(comp_id=surr_id, variable="V").squeeze()

    configs.append(
        {
            "data": [orig_v, surr_v],
            "legend": ["orig V", "surr V"],
            "colors": ["blue", "red"],
            "ylabel": "V [mV]",
            "linestyle": ["-", "--"],  # 重なった時に見やすいように破線にする
        }
    )
    # preprocessedとsurrogateのgate変数の時間変化
    # PCAで抽出されたTarget(教師データ)と、SINDyが予測した軌道を重ねて比較
    # preprocessed の featuresインデックスから "V" 以外の変数をすべて取得 (latent1, latent2...)
    prep_vars = preprocessed.get_index("features").get_level_values("variable").tolist()
    latent_vars = [v for v in prep_vars if v != "V"]

    for latent in latent_vars:
        prep_gate = preprocessed.sel(variable=latent).squeeze()
        surr_gate = surrogate["vars"].sel(comp_id=surr_id, variable=latent).squeeze()

        configs.append(
            {
                "data": [prep_gate, surr_gate],
                "legend": [f"target {latent}", f"surr {latent}"],
                "colors": ["blue", "red"],
                "ylabel": f"{latent}",
                "linestyle": ["-", "--"],
            }
        )
    # originalのgateの時間変化
    orig_gates = original["vars"].sel(comp_id=surr_id, gate=True)
    gate_names = orig_gates.coords["variable"].values.tolist()

    configs.append(
        {
            "data": [orig_gates.sel(variable=name).squeeze() for name in gate_names],
            "legend": gate_names,
            "ylabel": "orig gates",
            # 色はお好みで、draw_engineのデフォルトに任せるか指定するか
            "colors": ["green", "orange", "purple"][: len(gate_names)],
        }
    )

    return draw_engine(configs)
