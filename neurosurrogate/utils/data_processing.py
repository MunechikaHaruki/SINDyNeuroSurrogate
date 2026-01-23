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


def _create_xr(
    time,
    model_type: str,
    surrogate: bool = False,
    custom_features=None,
    params_dict={},
):
    if custom_features is not None:
        features = custom_features
    else:
        if surrogate is False:
            FEATURES_DICT = MODEL_FEATURES
        elif surrogate is True:
            FEATURES_DICT = SURROGATE_FEATURES
        features = FEATURES_DICT[model_type]

    attrs = {
        "model_type": model_type,
        "surrogate": surrogate,
        "gate_features": GATE_VARS[model_type],
        "params": params_dict,
        "dt": time[1] - time[0],
    }

    shape_vars = (len(time), len(features))
    shape_u = (len(time),)
    return xr.Dataset(
        {
            "vars": (
                ("time", "features"),
                np.empty(shape_vars),
            ),
            "I_ext": (("time"), np.empty(shape_u)),
        },
        coords={
            "time": time,
            "features": features,
        },
        attrs=attrs,
    )


def preprocess_dataset(
    model_type: str, i_ext, results, params: Dict, dt, surrogate=False
):
    time_array = np.arange(len(i_ext)) * dt
    # The FEATURES logic is moved to _create_xr
    dataset = _create_xr(
        time_array, model_type=model_type, surrogate=surrogate, params_dict=params
    )
    dataset["vars"].data = results
    dataset["I_ext"].data = i_ext
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


def _get_control_input(train_xr_dataset, data_type, direct=False):
    if data_type == "hh3" and direct is True:
        return train_xr_dataset["I_internal"].sel(direction="soma").to_numpy()
    else:
        return train_xr_dataset["I_ext"].to_numpy()
