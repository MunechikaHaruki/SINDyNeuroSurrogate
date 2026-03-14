from typing import Dict

import hydra
import mlflow
from prefect import flow, get_run_logger

from neurosurrogate.modeling import analyze_eval_results
from neurosurrogate.modeling.calc_engine import unified_simulater
from neurosurrogate.utils.plots import plot_simple


def train_model(surrogate, train_ds, target_comp_id):
    surrogate.fit(train_ds, target_comp_id)
    # surrogateモデルのロギング
    summary = surrogate.get_loggable_summary()

    mlflow.log_metrics(summary["metrics"])
    mlflow.log_params(summary["params"])

    for filename, content in summary["artifacts"]["texts"].items():
        mlflow.log_text(content, artifact_file=filename)

    for filename, fig in summary["artifacts"]["figures"].items():
        mlflow.log_figure(fig, artifact_file=filename)


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
    eval_result = analyze_eval_results(
        original_ds, predict_result, name, target_comp_id, surrogate_model
    )
    mlflow.log_metrics(eval_result["metrics"])
    for path, fig in eval_result["figures"].items():
        mlflow.log_figure(fig, artifact_file=path)


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
