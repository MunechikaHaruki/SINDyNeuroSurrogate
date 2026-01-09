import numpy as np
import xarray as xr
from loguru import logger

GATE_VAR_SLICE = slice(1, 4, None)
V_VAR_SLICE = slice(0, 1, None)


def get_gate_data(xr_dataset):
    return xr_dataset["vars"].to_numpy()[:, GATE_VAR_SLICE]


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
    transformed_xr = xr_data.copy().drop_vars("vars").drop_vars("features")
    transformed_xr["vars"] = xr.DataArray(
        new_vars,
        coords={
            "time": xr_data.coords["time"],
            "features": new_feature_names,
        },
        dims=["time", "features"],
    )
    return transformed_xr


def _prepare_train_data(train_xr_dataset, preprocessor):
    train_gate_data = get_gate_data(train_xr_dataset)
    V_data = train_xr_dataset["vars"].to_numpy()[:, V_VAR_SLICE]

    logger.info("Transforming training dataset...")
    transformed_gate = preprocessor.transform(train_gate_data)
    train = np.concatenate((V_data, transformed_gate), axis=1)
    logger.debug(train)
    return train


def _get_control_input(train_xr_dataset, model_cfg):
    if model_cfg.sel_train_u == "I_ext":
        return train_xr_dataset["I_ext"].to_numpy()
    elif model_cfg.sel_train_u == "soma":
        return train_xr_dataset["I_internal"].sel(direction="soma").to_numpy()
    raise ValueError(f"Invalid sel_train_u configuration: {model_cfg.sel_train_u}")
