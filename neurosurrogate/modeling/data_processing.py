from typing import Any, Dict

import numpy as np
import xarray as xr

GATE_VARS: Dict[str, list[str]] = {
    "hh": ["M", "H", "N"],
    "hh3": ["M", "H", "N"],
}


MODEL_FEATURES: Dict[str, Dict[str, Any]] = {
    "hh": ["V", "M", "H", "N"],
    "hh3": ["V", "M", "H", "N", "V_pre", "V_post"],
}

SURROGATE_FEATURES = {
    "hh": ["V", "latent1"],
    "hh3": ["V", "latent1", "V_pre", "V_post"],
}


def preprocess_dataset(
    model_type: str, i_ext, results, params: Dict, dt, surrogate=False
):
    if surrogate is False:
        FEATURES_DICT = MODEL_FEATURES
    elif surrogate is True:
        FEATURES_DICT = SURROGATE_FEATURES
    features = FEATURES_DICT[model_type]

    attrs = {
        "model_type": model_type,
        "surrogate": surrogate,
        "gate_features": GATE_VARS[model_type],
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
            dataset["vars"].sel(features="V_pre") - dataset["vars"].sel(features="V")
        )
        I_post = params["G_23"] * (
            dataset["vars"].sel(features="V") - dataset["vars"].sel(features="V_post")
        )
        I_soma = I_pre - I_post

        dataset["I_internal"] = xr.concat(
            [I_pre, I_post, I_soma], dim="direction"
        ).assign_coords(direction=["pre", "post", "soma"])

    return dataset
