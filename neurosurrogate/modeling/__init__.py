import logging

import numpy as np
from numba import types
from numba.typed import Dict
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

from ._simulater import (
    HH_Params_numba,
    ThreeComp_Params_numba,
    hh3_simulate_numba,
    hh_simulate_numba,
)
from ._surrogate import simulate_sindy, simulate_three_comp_numba
from .data_processing import (
    _create_xr,
    preprocess_dataset,
)

logger = logging.getLogger(__name__)

PARAMS_REGISTRY = {
    "hh": HH_Params_numba,
    "hh3": ThreeComp_Params_numba,
}
SIMULATOR_REGISTRY = {
    "hh": hh_simulate_numba,
    "hh3": hh3_simulate_numba,
}
SURROGATER_REGISTRY = {"hh": simulate_sindy, "hh3": simulate_three_comp_numba}


def instantiate_OmegaConf_params(cfg, data_type):
    if cfg is None:
        py_dict = {}
    else:
        py_dict = OmegaConf.to_container(cfg, resolve=True)
    nb_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    for k, v in py_dict.items():
        nb_dict[k] = float(v)
    return PARAMS_REGISTRY[data_type](nb_dict)


def simulater(
    neuron_cfg,
    data_type,
    DT,
    i_ext,
):
    params = instantiate_OmegaConf_params(neuron_cfg, data_type=data_type)
    results = SIMULATOR_REGISTRY[data_type](i_ext, params, DT)
    # Preprocess the simulation data
    return preprocess_dataset(
        model_type=data_type,
        i_ext=i_ext,
        results=results,
        params=neuron_cfg,
        dt=DT,
        surrogate=False,
    )


class PCAPreProcessorWrapper:
    def __init__(self):
        self.pca = PCA(n_components=1)

    def fit(self, train_xr_dataset):
        gate_features = train_xr_dataset.attrs["gate_features"]
        train_gate_data = (
            train_xr_dataset["vars"].sel(features=gate_features).to_numpy()
        )
        logger.info("Fitting preprocessor...")
        self.pca.fit(train_gate_data)

    def transform(self, xr_data):
        logger.critical(type(xr_data))
        gate_features = xr_data.attrs["gate_features"]
        xr_gate = xr_data["vars"].sel(features=gate_features).to_numpy()
        transformed_gate = self.pca.transform(xr_gate)
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


class SINDySurrogateWrapper:
    def __init__(self, cfg, preprocessor):
        self.cfg = cfg
        self.preprocessor = preprocessor

    def _prepare_train_data(self, train_xr_dataset):
        data_type = train_xr_dataset.attrs["model_type"]
        train = self.preprocessor.transform(train_xr_dataset)["vars"].to_numpy()
        if data_type == "hh3" and self.cfg.direct is True:
            u = train_xr_dataset["I_internal"].sel(direction="soma").to_numpy()
        else:
            u = train_xr_dataset["I_ext"].to_numpy()
        return train, u

    def fit(self, train_xr_dataset):
        # fit
        from neurosurrogate.utils.base_hh import hh_sindy, input_features

        self.sindy = hh_sindy
        train, u = self._prepare_train_data(train_xr_dataset)
        hh_sindy.fit(
            train,
            u=u,
            t=train_xr_dataset["time"].to_numpy(),
            feature_names=input_features,
        )

    def predict(self, init, dt, u, data_type, params_dict):
        logger.info(f"{data_type}のサロゲートモデルをテスト")
        params = instantiate_OmegaConf_params(params_dict, data_type=data_type)
        var = SURROGATER_REGISTRY[data_type](
            init=init,
            u=u,
            xi_matrix=self.sindy.coefficients(),
            params=params,
            dt=dt,
        )
        logger.critical(f"{var.shape}")
        sindy_result = preprocess_dataset(
            model_type=data_type,
            i_ext=u,
            results=var,
            params=params_dict,
            dt=dt,
            surrogate=True,
        )
        return sindy_result

    def eval(self, test_preprocessed_ds):
        return self.predict(
            init=test_preprocessed_ds["vars"][0].to_numpy(),
            dt=float(test_preprocessed_ds.attrs["dt"]),
            u=test_preprocessed_ds["I_ext"].to_numpy(),
            data_type=test_preprocessed_ds.attrs["model_type"],
            params_dict=test_preprocessed_ds.attrs["params"],
        )

    def get_loggable_summary(self) -> dict:
        return {
            "equations": self.sindy.equations(precision=3),
            "coefficients": self.sindy.optimizer.coef_,
            "feature_names": self.sindy.get_feature_names(),
            "model_params": str(self.sindy.optimizer.get_params),
        }
