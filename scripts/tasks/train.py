import mlflow
import numpy as np
from loguru import logger
from prefect import task

from neurosurrogate.modeling import SINDySurrogateWrapper


@task
def train_preprocessor(train_xr_dataset):
    """前処理器（Preprocessor）の学習を行うタスク"""
    from sklearn.decomposition import PCA

    preprocessor = PCA(n_components=1)
    gate_features = train_xr_dataset.attrs["gate_features"]
    train_gate_data = train_xr_dataset["vars"].sel(features=gate_features).to_numpy()
    logger.info("Fitting preprocessor...")
    preprocessor.fit(train_gate_data)
    return preprocessor


@task
def train_model(train_xr_dataset, preprocessor, surrogate_model_cfg):
    """モデルの学習を行うタスク"""

    surrogater = SINDySurrogateWrapper(surrogate_model_cfg, preprocessor)
    surrogater.fit(train_xr_dataset)

    logger.debug(f"train_dataset {train_xr_dataset}")
    logger.info("Fitting surrogate model...")
    return surrogater


@task
def log_train_model(surrogate):
    summary = surrogate.get_loggable_summary()
    mlflow.log_dict(
        summary["equations"],
        artifact_file="sindy_equations.txt",
    )
    mlflow.log_text(
        np.array2string(summary["coefficients"], precision=3),
        artifact_file="coef.txt",
    )
    feature_names = summary["feature_names"]
    mlflow.log_text("\n".join(feature_names), artifact_file="feature_names.txt")
    mlflow.log_param(
        "model_params",
        summary["model_params"],
    )


def train_flow(cfg, train_ds):
    # 2. Train Preprocessor
    preprocessor = train_preprocessor(train_xr_dataset=train_ds)

    # 3. Train Model
    surrogate_model = train_model(
        train_xr_dataset=train_ds,
        preprocessor=preprocessor,
        surrogate_model_cfg=cfg.models["surrogate"],
    )
    log_train_model(surrogate=surrogate_model)
    return preprocessor, surrogate_model
