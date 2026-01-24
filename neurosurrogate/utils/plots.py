# mypy: ignore-errors
import xarray as xr
from matplotlib.figure import Figure


def draw_time_series(plot_configs):
    # 描画処理
    n_rows = len(plot_configs)
    fig = Figure(figsize=(8, 2 * n_rows))
    axs = fig.subplots(nrows=n_rows, ncols=1, sharex=True)  # sharex=Trueを推奨

    for i, config in enumerate(plot_configs):
        ax = axs[i]
        d = config["data"]

        if isinstance(d, list):
            for line, lbl in zip(d, config.get("legend", [])):
                ax.plot(line.time, line, label=lbl)
        else:
            ax.plot(d.time, d)

        ax.set_ylabel(config["ylabel"])

        leg = config.get("legend")
        if leg is not None:
            ax.legend()
        if config.get("xlabel"):
            ax.set_xlabel(config["xlabel"])
    fig.tight_layout()

    return fig


def plot_simple(xr):
    # 属性の取得
    surrogate = xr.attrs.get("surrogate", False)
    model_type = xr.attrs.get("model_type", "hh")
    data = xr["vars"]
    i_ext = xr["I_ext"]

    # プロット構成の決定
    plot_configs = []

    # 1. 外部入力 (共通)
    plot_configs.append({"data": i_ext, "ylabel": "I_ext(t)"})

    # 2. 内部電流 (hh3のみ)
    if model_type == "hh3":
        i_int = xr["I_internal"]
        plot_configs.append(
            {
                "data": [i_int.sel(direction=d) for d in i_int.direction.values],
                "legend": i_int.direction.values,
                "ylabel": "I_internal",
            }
        )

    # 3. 電位 (V)
    v_features = ["V_pre", "V", "V_post"] if model_type == "hh3" else ["V"]
    plot_configs.append(
        {
            "data": data.sel(features=v_features),
            "ylabel": "V(t)",
            "legend": v_features if len(v_features) > 1 else None,
        }
    )

    # 4. ゲート変数 / 潜在変数
    gate_features = ["latent1"] if surrogate else ["M", "H", "N"]
    plot_configs.append(
        {
            "data": data.sel(features=gate_features),
            "ylabel": "g'" if surrogate and model_type == "hh" else "gates",
            "legend": gate_features,
            "xlabel": "Time [ms]",  # 単位は統一することを推奨
        }
    )
    fig = draw_time_series(plot_configs)
    return fig


def plot_preprocessed(preprocessed_xr, axes=None):
    """
    xarrayデータから特徴量ごとの時系列プロットを作成する。

    Args:
        preprocessed_xr: プロット対象のxarray.Dataset
        axes: 描画先のAxes配列 (省略時は新規Figureを作成)
    Returns:
        Figureオブジェクト (axes経由で渡された場合はその親Figure)
    """
    external_input = preprocessed_xr["I_ext"].to_numpy()
    data_vars = preprocessed_xr["vars"]
    features = data_vars.features.values
    num_features = len(features)
    total_rows = 1 + num_features

    # --- 1. Figure/Axesの準備 (副作用の制御) ---
    if axes is None:
        # plt.subplotsではなくFigureを直接生成（plt.close不要の設計）
        fig = Figure(figsize=(10, 4 * total_rows), layout="constrained")
        axs = fig.subplots(nrows=total_rows, ncols=1, sharex=True)
    else:
        # 外部から渡された場合はそれを使う（副作用は呼び出し側に帰属）
        if len(axes) != total_rows:
            raise ValueError(f"axes needs length {total_rows}, but got {len(axes)}")
        axs = axes
        fig = axs[0].get_figure()

    # --- 2. 描画ロジック (データの可視化) ---
    # 外部入力
    axs[0].plot(external_input, label="I_ext(t)")
    axs[0].set_ylabel("I_ext(t)")

    # 各特徴量
    for i, feature_name in enumerate(features):
        ax = axs[i + 1]
        # xarrayのプロット機能をそのまま使う（座標やラベルを自動活用できるため）
        data_vars.sel(features=feature_name).plot(ax=ax)
        ax.set_title("")  # デフォルトのタイトルを消して整理
        ax.set_ylabel(feature_name)

    axs[-1].set_xlabel("Time step")
    fig.tight_layout()
    return fig


def plot_diff(original: xr.Dataset, surrogate: xr.Dataset):
    u = surrogate["I_ext"].to_numpy()
    original = original["vars"]
    surrogate = surrogate["vars"]

    num_features = len(original.features.values)
    fig = Figure(figsize=(10, 4 * (1 + num_features)))
    axs = fig.subplots(nrows=1 + 2 * num_features, ncols=1, sharex=False)

    # plot external_input (I_ext)
    axs[0].plot(u, label="I_ext(t)", color="gold")
    axs[0].set_ylabel("I_ext(t)")

    # 各 feature についてループ
    for i, feature in enumerate(original.features.values):
        # 1. 元のデータをプロット (引数 'oridginal' から)
        axs[2 * i + 1].plot(
            original.time,
            original.sel(features=feature),
            color="blue",
            label=f"Original {feature}",
        )
        axs[2 * i + 1].set_ylabel(feature)

        # 2. サロゲートモデルのデータをプロット (引数 'surrogate' から)
        #    surrogate も 'time' と 'features' の座標を持つと仮定
        axs[2 * i + 2].plot(
            surrogate.time,
            surrogate.sel(features=feature),
            color="red",
            label=f"Surrogate {feature}",
        )
        axs[2 * i + 2].set_ylabel(f"Surrogate {feature}")

    axs[-1].set_xlabel("Time step")
    fig.tight_layout()
    return fig
