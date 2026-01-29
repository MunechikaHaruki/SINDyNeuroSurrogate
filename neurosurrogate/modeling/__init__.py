import logging
from typing import Any

import numpy as np
import xarray as xr
from numba import types
from numba.typed import Dict
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

from ..utils.plots import plot_compartment_behavior, plot_diff, plot_simple
from ._simulater import (
    HH_Params_numba,
    ThreeComp_Params_numba,
    hh3_simulate_numba,
    hh_simulate_numba,
)
from ._surrogate import simulate_sindy, simulate_three_comp_numba

logger = logging.getLogger(__name__)

PARAMS_REGISTRY = {
    "hh": HH_Params_numba,
    "hh3": ThreeComp_Params_numba,
}
SIMULATOR_REGISTRY = {
    "hh": hh_simulate_numba,
    "hh3": hh3_simulate_numba,
}

SIMULATOR_FEATURES: Dict[str, Dict[str, Any]] = {
    "hh": ["V_soma", "M", "H", "N"],
    "hh3": ["V_soma", "M", "H", "N", "V_pre", "V_post"],
}

SURROGATER_REGISTRY = {"hh": simulate_sindy, "hh3": simulate_three_comp_numba}

SURROGATER_FEATURES = {
    "hh": ["V_soma", "latent1"],
    "hh3": ["V_soma", "latent1", "V_pre", "V_post"],
}


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
        features=SIMULATOR_FEATURES[data_type],
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
        V_data = xr_data["vars"].sel(features="V_soma").to_numpy().reshape(-1, 1)
        new_vars = np.concatenate((V_data, transformed_gate), axis=1)
        new_feature_names = ["V_soma"] + [
            f"latent{i + 1}" for i in range(transformed_gate.shape[1])
        ]

        return xr.DataArray(
            data=new_vars,
            dims=("time", "features"),
            coords={
                "time": xr_data.time,
                "features": new_feature_names,
            },
            name="vars",
            attrs=xr_data.attrs,
        )


class SINDySurrogateWrapper:
    def __init__(self, cfg, preprocessor):
        self.cfg = cfg
        self.preprocessor = preprocessor

    def _prepare_train_data(self, train_xr_dataset):
        self.train_dataarray = self.preprocessor.transform(train_xr_dataset)
        if self.cfg.direct is True:
            self.u_dataarray = train_xr_dataset["I_internal"].sel(direction="soma")
        else:
            self.u_dataarray = train_xr_dataset["I_ext"]
        return self.train_dataarray.to_numpy(), self.u_dataarray.to_numpy()

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
            features=SURROGATER_FEATURES[data_type],
            params=params_dict,
            dt=dt,
            surrogate=True,
        )
        return sindy_result

    def eval(self, original_ds):
        transformed_dataarray = self.preprocessor.transform(original_ds)
        predict_result = self.predict(
            init=transformed_dataarray[0].to_numpy(),
            dt=float(original_ds.attrs["dt"]),
            u=original_ds["I_ext"].to_numpy(),
            data_type=original_ds.attrs["model_type"],
            params_dict=original_ds.attrs["params"],
        )

        if original_ds.attrs["model_type"] == "hh":
            u_inj = original_ds["I_ext"].to_numpy()
        elif original_ds.attrs["model_type"] == "hh3":
            u_inj = original_ds["I_internal"].sel(direction="soma")

        return {
            "surrogate_figure": plot_simple(predict_result),
            "diff": plot_diff(
                original=original_ds,
                preprocessed=transformed_dataarray,
                surrogate=predict_result,
            ),
            "preprocessed": plot_compartment_behavior(
                u=u_inj, xarray=transformed_dataarray
            ),
        }

    def get_loggable_summary(self) -> dict:
        return {
            "equations": self.sindy.equations(precision=3),
            "coefficients": self.sindy.optimizer.coef_,
            "feature_names": self.sindy.get_feature_names(),
            "model_params": str(self.sindy.optimizer.get_params),
            "train_figure": plot_compartment_behavior(
                xarray=self.train_dataarray, u=self.u_dataarray
            ),
        }
