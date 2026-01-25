from typing import Dict

import numpy as np
import xarray as xr


def preprocess_dataset(
    model_type: str,
    i_ext,
    results,
    features,
    params: Dict,
    dt,
    surrogate=False,
):
    attrs = {
        "model_type": model_type,
        "surrogate": surrogate,
        "gate_features": ["M", "H", "N"],
        "params": params,
        "dt": dt,
    }
    dataset = xr.Dataset(
        {
            "vars": (
                ("time", "features"),
                results,
            ),
            "I_ext": (("time"), i_ext),
        },
        coords={
            "time": np.arange(len(i_ext)) * dt,
            "features": features,
        },
        attrs=attrs,
    )

    if model_type == "hh3":
        I_pre = params["G_12"] * (
            dataset["vars"].sel(features="V_pre")
            - dataset["vars"].sel(features="V_soma")
        )
        I_post = params["G_23"] * (
            dataset["vars"].sel(features="V_soma")
            - dataset["vars"].sel(features="V_post")
        )
        I_soma = I_pre - I_post

        dataset["I_internal"] = xr.concat(
            [I_pre, I_post, I_soma], dim="direction"
        ).assign_coords(direction=["pre", "post", "soma"])

    return dataset
