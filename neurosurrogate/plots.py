# mypy: ignore-errors

import matplotlib.pyplot as plt
import xarray as xr


class MultiCompartmentDataVisualizer:
    def __init__(self, mc_xarray: xr.Dataset, figsize=(10, 12)):
        self.xr = mc_xarray
        self.figsize = figsize

    def plotCompartment(self, compartment_ind=6):
        data_vars = self.xr["vars"].sel(compartments=compartment_ind)
        i_ext = self.xr["I_ext"].sel(compartments=compartment_ind)
        fig, axs = plt.subplots(3, 1, figsize=self.figsize, sharex=True)

        axs[0].plot(i_ext.time, i_ext, label="I_ext(t)")
        axs[0].set_ylabel("I_ext(t)")
        axs[0].legend()

        axs[1].plot(data_vars.time, data_vars.sel(features="V"), label="V(t)")
        axs[1].set_ylabel("V(t)")
        axs[1].legend()

        all_features = self.xr["vars"].coords["features"].values
        gate_features = [f for f in all_features if f not in ["V", "XI"]]
        axs[2].plot(data_vars.time, data_vars.sel(features=gate_features))
        axs[2].legend(gate_features)
        axs[2].set_ylabel("gates")
        axs[2].set_xlabel("Time step")

        return fig

    def plotDataset(self):
        data_array = self.xr["vars"]
        time_num, feature_num, comp_num = data_array.shape
        data = data_array.to_numpy()
        reshaped = data.reshape(time_num, feature_num * comp_num)
        fig, axs = plt.subplots(1, 1, figsize=self.figsize)
        axs.plot(reshaped[:, :])
        axs.legend()
        return fig

    def plotAttr(self, attr_name="V"):
        data = self.xr["vars"].sel(features=attr_name)
        fig, axs = plt.subplots(3, 1, figsize=self.figsize)

        axs[0].plot(data.time, data[:, :5])
        axs[0].set_title(f"{attr_name} - first 5 compartments")
        axs[0].legend([f"{attr_name} {i}" for i in range(5)])

        axs[1].plot(data.time, data[:, 5:12])
        axs[1].set_title(f"{attr_name} - compartments 5 to 11")
        axs[1].legend([f"{attr_name} {i}" for i in range(5, 12)])

        axs[2].plot(data.time, data[:, 12:])
        axs[2].set_title(f"{attr_name} - compartments 12 and beyond")
        axs[2].legend([f"{attr_name} {i}" for i in range(12, data.shape[1])])
        return fig


def plot_hh(xr, figsize=(10, 8), surrogate=False):
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=False)
    data = xr["vars"]
    i_ext = xr["I_ext"]

    axs[0].plot(i_ext.time, i_ext, label="I_ext(t)")
    axs[0].set_ylabel("I_ext(t)")
    axs[0].legend()

    axs[1].plot(data.time, data.sel(features="V"), label="V(t)")
    axs[1].set_ylabel("V(t)")
    axs[1].legend()

    if not surrogate:
        axs[2].plot(data.time, data.sel(features=["M", "H", "N"]))
        axs[2].legend(["M", "H", "N"])
        axs[2].set_ylabel("gates")
        axs[2].set_xlabel("Time step")
    else:
        axs[2].plot(data.time, data.sel(features=["latent1"]))
        axs[2].legend(["latent1"])
        axs[2].set_ylabel("gates")
        axs[2].set_xlabel("Time step")

    return fig


def plot_3comp_hh(xr, figsize=(10, 8), surrogate=False):
    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=False)
    data = xr["vars"]
    i_ext = xr["I_ext"]
    i_ext_internal = xr["I_internal"]

    axs[0].plot(i_ext.time, i_ext, label="I_ext(t)")
    axs[0].set_ylabel("I_ext(t)")
    axs[0].legend()

    axs[1].plot(i_ext_internal.time, i_ext_internal.sel(direction="pre"), label="pre")
    axs[1].plot(i_ext_internal.time, i_ext_internal.sel(direction="soma"), label="soma")
    axs[1].plot(i_ext_internal.time, i_ext_internal.sel(direction="post"), label="post")
    axs[1].legend()
    axs[1].set_ylabel("I_internal")

    axs[2].plot(data.time, data.sel(features=["V_pre", "V", "V_post"]))
    axs[2].legend(["V_dend(t)", "V_soma(t)", "V_axon(t)"])
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
