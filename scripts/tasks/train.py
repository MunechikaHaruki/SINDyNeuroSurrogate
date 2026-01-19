import mlflow
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from prefect import task
from prefect.tasks import task_input_hash

from neurosurrogate.utils.data_processing import (
    _get_control_input,
    _prepare_train_data,
    get_gate_data,
    transform_dataset_with_preprocessor,
)
from neurosurrogate.utils.plots import _create_figure

from .utils import fig_to_buff, recursive_to_dict


@task
def train_preprocessor(train_xr_dataset):
    """前処理器（Preprocessor）の学習を行うタスク"""
    from sklearn.decomposition import PCA

    preprocessor = PCA(n_components=1)

    train_gate_data = get_gate_data(train_xr_dataset)

    logger.info("Fitting preprocessor...")
    preprocessor.fit(train_gate_data)
    return preprocessor


@task
def train_model(
    train_xr_dataset, preprocessor, surrogate_model_cfg, train_dataset_type
):
    """モデルの学習を行うタスク"""
    surrogate_model_cfg = OmegaConf.create(recursive_to_dict(surrogate_model_cfg))
    from neurosurrogate.modeling.surrogate import SINDySurrogate
    from neurosurrogate.utils.base_hh import hh_sindy, input_features

    train = _prepare_train_data(train_xr_dataset, preprocessor)
    u = _get_control_input(train_xr_dataset, data_type=train_dataset_type)
    hh_sindy.fit(
        train,
        u=u,
        t=train_xr_dataset["time"].to_numpy(),
        feature_names=input_features,
    )
    surrogate = SINDySurrogate(hh_sindy, params=surrogate_model_cfg["params"])
    logger.debug(f"train_dataset {train_xr_dataset}")
    logger.info("Fitting surrogate model...")
    return surrogate


@task
def log_train_model(surrogate):
    mlflow.log_dict(
        surrogate.sindy.equations(precision=3),
        artifact_file="sindy_equations.txt",
    )
    mlflow.log_text(
        np.array2string(surrogate.sindy.optimizer.coef_, precision=3),
        artifact_file="coef.txt",
    )
    feature_names = surrogate.sindy.get_feature_names()
    mlflow.log_text("\n".join(feature_names), artifact_file="feature_names.txt")
    mlflow.log_param("sindy_params", str(surrogate.sindy.optimizer.get_params))


@task(cache_key_fn=task_input_hash, persist_result=True)
def preprocess_single_data(dataset_name, preprocessor, xr_data):
    transformed_xr = transform_dataset_with_preprocessor(xr_data, preprocessor)
    logger.info(f"Transformed xr dataset: {dataset_name}")
    return transformed_xr


@task(cache_key_fn=task_input_hash, persist_result=True)
def log_single_preprocess_data(dataset_key, dataset_type, xr_data):
    """1つのデータセットに対して処理とログ出力を行う"""
    external_input = _get_control_input(xr_data, dataset_type)
    fig = _create_figure(xr_data["vars"], external_input)
    return fig_to_buff(fig)
