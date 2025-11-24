from typing import Any, Dict

import h5py
import numpy as np
import xarray as xr

from neurosurrogate.config import INTERIM_DATA_DIR, RAW_DATA_DIR


def preprocess_dataset(model_type: str, file_name: str, params):
    MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
        "hh": {
            "features": ["V", "M", "H", "N"],
            "dims": (["time", "features"], ["time"]),
            "compartments": False,
        },
        "hh3": {
            "features": ["V", "M", "H", "N", "V_pre", "V_post"],
            "dims": (["time", "features"], ["time"]),
            "compartments": False,
        },
        "traub": {
            "features": ["V", "XI", "M", "S", "N", "C", "A", "H", "R", "B", "Q"],
            "dims": (["time", "features", "compartments"], ["time", "compartments"]),
            "compartments": True,
        },
    }
    config = MODEL_CONFIG[model_type]

    with h5py.File(RAW_DATA_DIR / model_type / file_name, "r") as f:
        coords: Dict[str, Any] = {
            "time": f["time"],
            "features": config["features"],
        }
        if config["compartments"]:
            coords["compartments"] = np.arange(f["vars"].shape[-1])

        dataset = xr.Dataset(
            {
                "vars": (config["dims"][0], f["vars"]),
                "I_ext": (config["dims"][1], f["I_ext"]),
            },
            coords=coords,
        )

        if model_type == "hh3":
            I_pre = params.g_12 * (
                dataset["vars"].sel(features="V_pre")
                - dataset["vars"].sel(features="V")
            )
            I_post = params.g_23 * (
                dataset["vars"].sel(features="V")
                - dataset["vars"].sel(features="V_post")
            )
            I_soma = I_pre - I_post

            dataset["I_internal"] = xr.concat(
                [I_pre, I_post, I_soma], dim="direction"
            ).assign_coords(direction=["pre", "post", "soma"])

    netcdf_path = INTERIM_DATA_DIR / model_type / file_name.replace(".h5", ".nc")
    dataset.to_netcdf(netcdf_path)
    return netcdf_path
