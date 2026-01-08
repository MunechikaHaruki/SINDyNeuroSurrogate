import numpy as np
from loguru import logger

GATE_VAR_SLICE = slice(1, 4, None)
V_VAR_SLICE = slice(0, 1, None)


def _prepare_train_data(train_xr_dataset, preprocessor):
    train_gate_data = train_xr_dataset["vars"].to_numpy()[:, GATE_VAR_SLICE]
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
