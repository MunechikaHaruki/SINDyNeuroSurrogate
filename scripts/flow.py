import logging
from typing import Dict

import hydra
import mlflow

# from prefect import flow, get_run_logger, task
from neurosurrogate.modeling import analyze_eval_results
from neurosurrogate.modeling.calc_engine import unified_simulater

logger = logging.getLogger(__name__)


@mlflow.trace
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


@mlflow.trace
def generate_dataset_flow(dataset_key, datasets_cfg, models_arch):
    dataset_cfg = datasets_cfg[dataset_key]
    data_type = dataset_cfg["data_type"]

    ds = unified_simulater(
        u=hydra.utils.instantiate(dataset_cfg["current"]),
        dt=dataset_cfg["dt"],
        net=models_arch[data_type],
    )
    ds.attrs["model_type"] = data_type
    return ds


@mlflow.trace
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
    summary = analyze_eval_results(
        original_ds, predict_result, target_comp_id, surrogate_model
    )
    mlflow.log_metrics(summary["metrics"])
    for filename, content in summary["artifacts"]["texts"].items():
        mlflow.log_text(content, artifact_file=filename)

    for filename, fig in summary["artifacts"]["figures"].items():
        mlflow.log_figure(fig, artifact_file=filename)


def main_flow(datasets_cfg: Dict, surrogate_model, models_arch, run_name):
    logger.info("Start Flow:start generate train data")
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"run_id:{run.info.run_id}")
        mlflow.log_dict(datasets_cfg, "datasets.yaml")
        train_ds = generate_dataset_flow("train", datasets_cfg, models_arch)
        target_comp_id = datasets_cfg["train"]["target_comp_id"]
        logger.info("Start Training")
        train_model(surrogate_model, train_ds, target_comp_id)
        for key in datasets_cfg.keys():
            logger.info(f"start {key}'s evaluation")
            with mlflow.start_run(run_name=f"Eval_{key}", nested=True):
                mlflow.set_tag("eval_dataset", key)
                mlflow.log_dict(datasets_cfg[key], "dataset.yaml")
                ds = generate_dataset_flow(key, datasets_cfg, models_arch)
                eval_diff(ds, key, datasets_cfg, surrogate_model, models_arch)
