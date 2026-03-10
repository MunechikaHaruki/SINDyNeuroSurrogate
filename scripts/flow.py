from typing import Dict

import hydra
import mlflow
import numpy as np
from base import MC_MODELS, SURROGATE_TARGET, SINDY_MODEl
from prefect import flow, get_run_logger, task

from neurosurrogate.modeling import (
    SINDySurrogateWrapper,
)
from neurosurrogate.modeling.calc_engine import unified_simulater
from neurosurrogate.utils.plots import plot_compartment_behavior, plot_diff, plot_simple


def log_train_model(surrogate):
    summary = surrogate.get_loggable_summary()
    mlflow.log_dict(
        summary["equations"],
        artifact_file="sindy_equations.txt",
    )
    coef = summary["coefficients"]
    mlflow.log_text(
        np.array2string(coef, precision=3),
        artifact_file="coef.txt",
    )

    nonzero_term_num = np.count_nonzero(coef)
    mlflow.log_metrics(
        metrics={
            "nonzero_term_num": nonzero_term_num,
            "nonzero_term_ratio": nonzero_term_num / coef.size,
        }
    )

    mlflow.log_metrics(summary["static_calc_cost"])

    mlflow.log_text(
        "\n".join(summary["feature_names"]), artifact_file="feature_names.txt"
    )
    mlflow.log_text(
        "\n".join(summary["active_features"]), artifact_file="active_features.txt"
    )

    mlflow.log_param(
        "model_params",
        summary["model_params"],
    )
    mlflow.log_figure(summary["train_figure"], artifact_file="train.png")


def generate_dataset_flow(dataset_key, datasets_cfg):
    dataset_cfg = datasets_cfg[dataset_key]
    data_type = dataset_cfg["data_type"]

    ds = unified_simulater(
        u=hydra.utils.instantiate(
            dataset_cfg["current"], current_seed=dataset_cfg["seed"]
        ),
        dt=dataset_cfg["dt"],
        net=MC_MODELS[data_type],
    )
    ds.attrs["model_type"] = data_type
    fig = plot_simple(ds)
    mlflow.log_figure(fig, artifact_file=f"original/{data_type}/{dataset_key}.png")
    return ds


def eval_diff(surrogater, original_ds, name):
    data_type = original_ds.attrs["model_type"]
    predict_result = unified_simulater(
        dt=float(original_ds.attrs["dt"]),
        u=original_ds["I_ext"].to_numpy(),
        net=MC_MODELS[data_type],
        surrogate_target=SURROGATE_TARGET[data_type],
        surrogate_model=surrogater,
    )

    target_comp_id = SURROGATE_TARGET[data_type]

    transformed_dataarray = surrogater.preprocessor.transform(
        original_ds, target_comp_id=target_comp_id
    )

    mlflow.log_figure(
        plot_compartment_behavior(
            u=original_ds["I_internal"].sel(node_id=target_comp_id),
            xarray=transformed_dataarray,
        ),
        artifact_file=f"preprocessed/{data_type}/{name}.png",
    )
    mlflow.log_figure(
        plot_simple(predict_result),
        artifact_file=f"surrogate/{data_type}/{name}.png",
    )
    mlflow.log_figure(
        plot_diff(
            original=original_ds,
            preprocessed=transformed_dataarray,
            surrogate=predict_result,
        ),
        artifact_file=f"compare/{data_type}/{name}.png",
    )


@task
def train_task(train_ds):

    # 3. Train Model
    surrogate_model = SINDySurrogateWrapper(SINDY_MODEl[0], SINDY_MODEl[1])
    target_comp_id = SURROGATE_TARGET[train_ds.attrs["model_type"]]

    surrogate_model.fit(train_ds, target_comp_id=target_comp_id)
    return surrogate_model


@flow
def main_flow(datasets_cfg: Dict):
    logger = get_run_logger()
    logger.info("Start Flow")

    logger.info("start generate train data")
    train_ds = generate_dataset_flow("train", datasets_cfg)
    surrogate_model = train_task(train_ds)
    log_train_model(surrogate_model)

    for name in datasets_cfg.keys():
        logger.info(f"start {name}'s evaluation")
        ds = generate_dataset_flow(name, datasets_cfg)
        eval_diff(surrogate_model, ds, name)
