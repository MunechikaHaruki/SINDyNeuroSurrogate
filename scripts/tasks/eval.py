from loguru import logger
from prefect import task

from neurosurrogate.utils import PLOTTER_REGISTRY
from neurosurrogate.utils.data_processing import (
    _get_control_input,
    transform_dataset_with_preprocessor,
)
from neurosurrogate.utils.plots import _create_figure, plot_diff

from .data import generate_dataset_flow
from .utils import fig_to_buff, log_plot_to_mlflow


@task
def single_eval(preprocessed_ds, surrogate_model):
    logger.info(f"{preprocessed_ds} started to process")
    prediction = surrogate_model.eval(preprocessed_ds)
    logger.info(f"prediction_result:{prediction}")

    logger.trace(prediction)
    return prediction


@task
def log_diff_eval(surrogate_result, preprocessed_result):
    u = preprocessed_result["I_ext"].to_numpy()
    fig = plot_diff(u, preprocessed_result["vars"], surrogate_result["vars"])
    return fig_to_buff(fig)


@task
def log_single_eval(data_type, surrogate_result):
    logger.critical("jfoisadjf")
    fig = PLOTTER_REGISTRY[data_type](
        surrogate_result,
        surrogate=True,
    )
    logger.critical("kljdgla")
    return fig_to_buff(fig)


@task
def preprocess_single_data(dataset_name, preprocessor, xr_data):
    transformed_xr = transform_dataset_with_preprocessor(xr_data, preprocessor)
    logger.info(f"Transformed xr dataset: {dataset_name}")
    return transformed_xr


@task
def log_single_preprocess_data(dataset_key, dataset_type, xr_data):
    """1つのデータセットに対して処理とログ出力を行う"""
    external_input = _get_control_input(xr_data, dataset_type)
    fig = _create_figure(xr_data["vars"], external_input)
    return fig_to_buff(fig)


def eval_flow(
    name: str,
    preprocessor,
    surrogate_model,
    cfg,
):
    dataset_cfg = cfg.datasets[name]
    data_type = dataset_cfg.data_type

    ds = generate_dataset_flow(name, cfg)

    transformed_ds = preprocess_single_data(
        dataset_name=name, preprocessor=preprocessor, xr_data=ds
    )
    log_plot_to_mlflow(
        log_single_preprocess_data(
            dataset_key=name,
            dataset_type=data_type,
            xr_data=transformed_ds,
        ),
        f"preprocessed/{data_type}/{name}.png",
    )

    eval_result = single_eval(
        preprocessed_ds=transformed_ds,
        surrogate_model=surrogate_model,
    )

    log_plot_to_mlflow(
        log_single_eval(data_type=data_type, surrogate_result=eval_result),
        f"surrogate/{data_type}/{name}.png",
    )
    log_plot_to_mlflow(
        log_diff_eval(
            surrogate_result=eval_result,
            preprocessed_result=transformed_ds,
        ),
        f"compare/{data_type}/{name}.png",
    )
