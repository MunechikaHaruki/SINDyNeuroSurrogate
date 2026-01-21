from loguru import logger
from prefect import task

from neurosurrogate.dataset_utils._base import calc_ThreeComp_internal
from neurosurrogate.utils import PLOTTER_REGISTRY
from neurosurrogate.utils.data_processing import _get_control_input
from neurosurrogate.utils.plots import plot_diff

from .utils import fig_to_buff


@task
def single_eval(data_type, params, preprocessed_ds, surrogate_model):
    logger.info(f"{preprocessed_ds} started to process")

    input_data = {
        "init": preprocessed_ds["vars"][0],
        "dt": 0.01,
        "iter": len(preprocessed_ds["time"].to_numpy()),
        "u": _get_control_input(preprocessed_ds, data_type=data_type),
        "data_type": data_type,
    }
    logger.info(f"input:{input_data}")

    prediction = surrogate_model.predict(**input_data)
    logger.info(f"prediction_result:{prediction}")
    if data_type == "hh3":
        calc_ThreeComp_internal(
            prediction,
            params.get("G_12"),
            params.get("G_23"),
        )

    logger.trace(prediction)
    return prediction


@task
def log_diff_eval(surrogate_result, preprocessed_result):
    u = preprocessed_result["I_ext"].to_numpy()
    fig = plot_diff(u, preprocessed_result["vars"], surrogate_result["vars"])
    return fig_to_buff(fig)


@task
def log_single_eval(data_type, surrogate_result):
    fig = PLOTTER_REGISTRY[data_type](
        surrogate_result,
        surrogate=True,
    )
    return fig_to_buff(fig)
