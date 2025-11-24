# mypy: ignore-errors

import os

import streamlit as st
import typer
import xarray as xr

from neurosurrogate.config import INTERIM_DATA_DIR
from neurosurrogate.plots import MultiCompartmentDataVisualizer, plot_3comp_hh, plot_hh

app = typer.Typer()


def StreamLitVisualizer():
    st.title("Data Visualizer")

    model_type = st.selectbox("Select model type", ["traub", "hh", "hh3"])

    files = [
        filename
        for filename in os.listdir(INTERIM_DATA_DIR / model_type)
        if filename.endswith(".nc")
    ]
    if not files:
        st.warning(f"No .h5 files found at {model_type}")
        return

    selected_file = st.selectbox("Select HDF5 file", files)

    if model_type == "traub":
        dataset = xr.open_dataset(INTERIM_DATA_DIR / model_type / selected_file)
        visualizer = MultiCompartmentDataVisualizer(dataset)

        plot_type = st.selectbox(
            "Select plot type", ["Compartment", "Dataset", "Attribute"]
        )

        if plot_type == "Compartment":
            compartment_ind = st.number_input("Compartment Index", min_value=0, value=6)
            fig = visualizer.plotCompartment(compartment_ind)
        elif plot_type == "Dataset":
            fig = visualizer.plotDataset()
        elif plot_type == "Attribute":
            attr_name = st.selectbox("Attribute Name", visualizer.xr["features"].values)
            fig = visualizer.plotAttr(attr_name)

    elif model_type == "hh":
        dataset = xr.open_dataset(INTERIM_DATA_DIR / model_type / selected_file)
        fig = plot_hh(dataset)
    elif model_type == "hh3":
        dataset = xr.open_dataset(INTERIM_DATA_DIR / model_type / selected_file)
        fig = plot_3comp_hh(dataset)
    st.pyplot(fig)


if __name__ == "__main__":
    StreamLitVisualizer()
