# mypy: ignore-errors
import xarray as xr
from matplotlib.figure import Figure


def plot_hh(xr, surrogate=False):
    fig = Figure()
    axs = fig.subplots(nrows=3, ncols=1, sharex=False)
    data = xr["vars"]
    i_ext = xr["I_ext"]

    axs[0].plot(i_ext.time, i_ext, label="I_ext(t)")
    axs[0].set_ylabel("I_ext(t)")

    axs[1].plot(data.time, data.sel(features="V"), label="V(t)")
    axs[1].set_ylabel("V(t)")

    if not surrogate:
        axs[2].plot(data.time, data.sel(features=["M", "H", "N"]))
        axs[2].legend(["M", "H", "N"])
        axs[2].set_ylabel("gates")
        axs[2].set_xlabel("Time step")
    else:
        axs[2].plot(data.time, data.sel(features=["latent1"]))
        axs[2].set_ylabel("g'")
        axs[2].set_xlabel("Time step")

    return fig


def plot_3comp_hh(xr, surrogate=False):
    fig = Figure()
    axs = fig.subplots(nrows=4, ncols=1, sharex=False)
    data = xr["vars"]
    i_ext = xr["I_ext"]
    i_ext_internal = xr["I_internal"]

    axs[0].plot(i_ext.time, i_ext, label="I_ext(t)")
    axs[0].set_ylabel("I_ext(t)")

    axs[1].plot(i_ext_internal.time, i_ext_internal.sel(direction="pre"), label="dend")
    axs[1].plot(i_ext_internal.time, i_ext_internal.sel(direction="soma"), label="soma")
    axs[1].plot(i_ext_internal.time, i_ext_internal.sel(direction="post"), label="axon")
    axs[1].legend()
    axs[1].set_ylabel("I_internal")

    axs[2].plot(data.time, data.sel(features=["V_pre", "V", "V_post"]))
    axs[2].legend()
    axs[2].set_ylabel("V(t)")

    if not surrogate:
        axs[3].plot(data.time, data.sel(features=["M", "H", "N"]))
        axs[3].legend(["M", "H", "N"])
        axs[3].set_ylabel("gates")
        axs[3].set_xlabel("Time [ms]")
    else:
        axs[3].plot(data.time, data.sel(features=["latent1"]))
        axs[3].legend(["latent1"])
        axs[3].set_ylabel("gates")
        axs[3].set_xlabel("Time [ms]")

    return fig


def create_preprocessed_figure(preprocessed_xr, axes=None):
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
    axs[0].legend(loc="upper right")

    # 各特徴量
    for i, feature_name in enumerate(features):
        ax = axs[i + 1]
        # xarrayのプロット機能をそのまま使う（座標やラベルを自動活用できるため）
        data_vars.sel(features=feature_name).plot(ax=ax)
        ax.set_title("")  # デフォルトのタイトルを消して整理
        ax.set_ylabel(feature_name)

    axs[-1].set_xlabel("Time step")

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
    fig.tight_layout()  # レイアウトを自動調整
    return fig
