from typing import Any, Dict

import numpy as np
import xarray as xr
from loguru import logger

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


def transform_dataset_with_preprocessor(xr_data, preprocessor):
    """
    Transforms the dataset using the preprocessor on the gate variables.

    Args:
        xr_data (xr.Dataset): The input xarray dataset.
        preprocessor: The fitted preprocessor (e.g., PCA or Autoencoder).

    Returns:
        xr.Dataset: A new xarray dataset with transformed variables.
    """
    gate_features = xr_data.attrs["gate_features"]
    xr_gate = xr_data["vars"].sel(features=gate_features).to_numpy()
    transformed_gate = preprocessor.transform(xr_gate)
    V_data = xr_data["vars"].sel(features="V").to_numpy().reshape(-1, 1)
    new_vars = np.concatenate((V_data, transformed_gate), axis=1)
    new_feature_names = ["V"] + [
        f"latent{i + 1}" for i in range(transformed_gate.shape[1])
    ]

    dataset = _create_xr(
        time=xr_data.coords["time"],
        model_type=xr_data.attrs.get("model_type"),
        custom_features=new_feature_names,
        params_dict=xr_data.attrs["params"],
    )
    dataset["vars"].data = new_vars
    dataset["I_ext"].data = xr_data["I_ext"].data
    return dataset


def _prepare_train_data(train_xr_dataset, preprocessor):
    gate_features = train_xr_dataset.attrs["gate_features"]
    train_gate_data = train_xr_dataset["vars"].sel(features=gate_features).to_numpy()
    V_data = train_xr_dataset["vars"].sel(features="V").to_numpy().reshape(-1, 1)

    logger.info("Transforming training dataset...")
    transformed_gate = preprocessor.transform(train_gate_data)
    train = np.concatenate((V_data, transformed_gate), axis=1)
    logger.debug(train)
    return train


def _get_control_input(train_xr_dataset, data_type, direct=False):
    if data_type == "hh3" and direct is True:
        return train_xr_dataset["I_internal"].sel(direction="soma").to_numpy()
    else:
        return train_xr_dataset["I_ext"].to_numpy()
