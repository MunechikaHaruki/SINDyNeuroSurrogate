from .plots import plot_3comp_hh, plot_hh

PLOTTER_REGISTRY = {"hh": plot_hh, "hh3": plot_3comp_hh}
