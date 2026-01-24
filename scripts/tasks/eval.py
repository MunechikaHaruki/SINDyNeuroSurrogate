import mlflow
from loguru import logger
from prefect import task

from neurosurrogate.utils import PLOTTER_REGISTRY
from neurosurrogate.utils.plots import create_preprocessed_figure, plot_diff

from .data import generate_dataset_flow


@task
def single_eval(preprocessed_ds, surrogate_model):
    logger.info(f"{preprocessed_ds} started to process")
    prediction = surrogate_model.eval(preprocessed_ds)
    logger.info(f"prediction_result:{prediction}")

    logger.trace(prediction)
    return prediction


def eval_flow(
    name: str,
    preprocessor,
    surrogate_model,
    cfg,
):
    dataset_cfg = cfg.datasets[name]
    data_type = dataset_cfg.data_type

    # generate_dataset
    ds = generate_dataset_flow(name, cfg)
    transformed_ds = task(preprocessor.transform)(ds)
    logger.info(f"Transformed xr: {name}")

    fig = task(create_preprocessed_figure)(transformed_ds)
    mlflow.log_figure(fig, artifact_file=f"preprocessed/{data_type}/{name}.png")

    eval_result = single_eval(
        preprocessed_ds=transformed_ds,
        surrogate_model=surrogate_model,
    )

    fig = PLOTTER_REGISTRY[data_type](
        eval_result,
        surrogate=True,
    )
    mlflow.log_figure(fig, artifact_file=f"surrogate/{data_type}/{name}.png")
    fig = plot_diff(transformed_ds, eval_result)
    mlflow.log_figure(fig, artifact_file=f"compare/{data_type}/{name}.png")
