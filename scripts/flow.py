import json
from typing import Dict

import hydra
import mlflow
import numpy as np
from prefect import flow, get_run_logger

from neurosurrogate.modeling.calc_engine import unified_simulater
from neurosurrogate.utils.plots import plot_compartment_behavior, plot_diff, plot_simple


def train_model(surrogate, train_ds, target_comp_id):
    surrogate.fit(train_ds, target_comp_id)
    # surrogateモデルのロギング
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

    mlflow.log_text(
        json.dumps(summary["feature_cost_map"], indent=2),
        artifact_file="feature_cost_map.txt",
    )

    mlflow.log_param(
        "model_params",
        summary["model_params"],
    )
    mlflow.log_figure(summary["train_figure"], artifact_file="train.png")


def generate_dataset_flow(dataset_key, datasets_cfg, models_arch):
    dataset_cfg = datasets_cfg[dataset_key]
    data_type = dataset_cfg["data_type"]

    ds = unified_simulater(
        u=hydra.utils.instantiate(dataset_cfg["current"]),
        dt=dataset_cfg["dt"],
        net=models_arch[data_type],
    )
    ds.attrs["model_type"] = data_type
    fig = plot_simple(ds)
    mlflow.log_figure(fig, artifact_file=f"original/{data_type}/{dataset_key}.png")
    return ds


def eval_diff(original_ds, name, datasets_cfg, surrogate_model, models_arch):
    data_type = original_ds.attrs["model_type"]
    target_comp_id = datasets_cfg[name]["target_comp_id"]
    predict_result = unified_simulater(
        dt=float(original_ds.attrs["dt"]),
        u=original_ds["I_ext"].to_numpy(),
        net=models_arch[data_type],
        surrogate_target=target_comp_id,
        surrogate_model=surrogate_model,
    )

    transformed_dataarray = surrogate_model.preprocessor.transform(
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


@flow
def main_flow(datasets_cfg: Dict, surrogate_model, models_arch):
    logger = get_run_logger()
    logger.info("Start Flow")
    logger.info("start generate train data")
    train_ds = generate_dataset_flow("train", datasets_cfg, models_arch)
    target_comp_id = datasets_cfg["train"]["target_comp_id"]
    logger.info("Start Training")
    train_model(surrogate_model, train_ds, target_comp_id)
    for key in datasets_cfg.keys():
        logger.info(f"start {key}'s evaluation")
        ds = generate_dataset_flow(key, datasets_cfg, models_arch)
        eval_diff(ds, key, datasets_cfg, surrogate_model, models_arch)
