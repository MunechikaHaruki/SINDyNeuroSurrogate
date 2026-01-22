from typing import Any, Dict

import numpy as np
import xarray as xr
from loguru import logger

GATE_VAR_SLICE = slice(1, 4, None)
V_VAR_SLICE = slice(0, 1, None)


def get_gate_data(xr_dataset):
    return xr_dataset["vars"].to_numpy()[:, GATE_VAR_SLICE]


def create_xr(var, time, u, features):
    return xr.Dataset(
        {
            "vars": (
                ("time", "features"),
                var,
            ),
            "I_ext": (("time"), u),
        },
        coords={
            "time": time,
            "features": features,
        },
    )


def transform_dataset_with_preprocessor(xr_data, preprocessor):
    """
    Transforms the dataset using the preprocessor on the gate variables.

    Args:
        xr_data (xr.Dataset): The input xarray dataset.
        preprocessor: The fitted preprocessor (e.g., PCA or Autoencoder).

    Returns:
        xr.Dataset: A new xarray dataset with transformed variables.
    """
    xr_gate = get_gate_data(xr_data)
    transformed_gate = preprocessor.transform(xr_gate)
    V_data = xr_data["vars"][:, V_VAR_SLICE].to_numpy().reshape(-1, 1)
    new_vars = np.concatenate((V_data, transformed_gate), axis=1)
    new_feature_names = ["V"] + [
        f"latent{i + 1}" for i in range(transformed_gate.shape[1])
    ]
    return create_xr(
        var=new_vars,
        time=xr_data.coords["time"],
        u=xr_data["I_ext"].data,
        features=new_feature_names,
    )


def _prepare_train_data(train_xr_dataset, preprocessor):
    train_gate_data = get_gate_data(train_xr_dataset)
    V_data = train_xr_dataset["vars"].to_numpy()[:, V_VAR_SLICE]

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


MODEL_FEATURES: Dict[str, Dict[str, Any]] = {
    "hh": ["V", "M", "H", "N"],
    "hh3": ["V", "M", "H", "N", "V_pre", "V_post"],
    "traub": ["V", "XI", "M", "S", "N", "C", "A", "H", "R", "B", "Q"],
}


def calc_ThreeComp_internal(dataset, G_12, G_23):
    I_pre = G_12 * (
        dataset["vars"].sel(features="V_pre") - dataset["vars"].sel(features="V")
    )
    I_post = G_23 * (
        dataset["vars"].sel(features="V") - dataset["vars"].sel(features="V_post")
    )
    I_soma = I_pre - I_post

    dataset["I_internal"] = xr.concat(
        [I_pre, I_post, I_soma], dim="direction"
    ).assign_coords(direction=["pre", "post", "soma"])


def preprocess_dataset(model_type: str, i_ext, results, params: Dict, time_array):
    dataset = create_xr(
        results, time_array, u=i_ext, features=MODEL_FEATURES[model_type]
    )

    if model_type == "hh3":
        calc_ThreeComp_internal(dataset, params["G_12"], params["G_23"])

    return dataset
