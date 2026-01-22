import mlflow
import numpy as np
from loguru import logger
from prefect import task

from neurosurrogate.utils.data_processing import (
    _get_control_input,
    _prepare_train_data,
    get_gate_data,
)


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
    from neurosurrogate.utils.base_hh import hh_sindy, input_features

    train = _prepare_train_data(train_xr_dataset, preprocessor)
    u = _get_control_input(train_xr_dataset, data_type=train_dataset_type)
    hh_sindy.fit(
        train,
        u=u,
        t=train_xr_dataset["time"].to_numpy(),
        feature_names=input_features,
    )
    logger.debug(f"train_dataset {train_xr_dataset}")
    logger.info("Fitting surrogate model...")
    return hh_sindy


@task
def log_train_model(surrogate):
    sindy = surrogate
    mlflow.log_dict(
        sindy.equations(precision=3),
        artifact_file="sindy_equations.txt",
    )
    mlflow.log_text(
        np.array2string(sindy.optimizer.coef_, precision=3),
        artifact_file="coef.txt",
    )
    feature_names = sindy.get_feature_names()
    mlflow.log_text("\n".join(feature_names), artifact_file="feature_names.txt")
    mlflow.log_param("sindy_params", str(sindy.optimizer.get_params))


def train_flow(cfg, train_ds):
    # 2. Train Preprocessor
    preprocessor = train_preprocessor(train_xr_dataset=train_ds)

    # 3. Train Model
    surrogate_model = train_model(
        train_xr_dataset=train_ds,
        preprocessor=preprocessor,
        surrogate_model_cfg=cfg.models["surrogate"],
        train_dataset_type=cfg.datasets["train"].data_type,
    )
    log_train_model(surrogate=surrogate_model)
    return preprocessor, surrogate_model
