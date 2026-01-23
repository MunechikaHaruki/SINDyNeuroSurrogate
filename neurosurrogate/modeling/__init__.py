from loguru import logger
from numba import types
from numba.typed import Dict
from omegaconf import OmegaConf

from ..utils.data_processing import (
    _get_control_input,
    _prepare_train_data,
    preprocess_dataset,
)
from ._simulater import (
    HH_Params_numba,
    ThreeComp_Params_numba,
    hh3_simulate_numba,
    hh_simulate_numba,
)
from ._surrogate import simulate_sindy, simulate_three_comp_numba

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


class SINDySurrogateWrapper:
    def __init__(self, cfg, preprocessor):
        self.cfg = cfg
        self.preprocessor = preprocessor

    def fit(self, train_xr_dataset):
        data_type = train_xr_dataset.attrs["model_type"]
        # fit
        from neurosurrogate.utils.base_hh import hh_sindy, input_features

        self.sindy = hh_sindy

        train = _prepare_train_data(train_xr_dataset, self.preprocessor)
        u = _get_control_input(train_xr_dataset, data_type=data_type)
        hh_sindy.fit(
            train,
            u=u,
            t=train_xr_dataset["time"].to_numpy(),
            feature_names=input_features,
        )

    def predict(self, init, dt, u, data_type, params_dict):
        print(f"{type(init)},{type(dt)},{type(u)},{type(data_type)}")
        logger.info(f"{data_type}のサロゲートモデルをテスト")
        logger.critical(type(u))
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
            u=_get_control_input(
                test_preprocessed_ds, data_type=test_preprocessed_ds.attrs["model_type"]
            ),
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
