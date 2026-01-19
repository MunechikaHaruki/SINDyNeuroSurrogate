import tempfile

import matplotlib.pyplot as plt
import mlflow
from loguru import logger
from prefect import task

from neurosurrogate.dataset_utils._base import calc_ThreeComp_internal
from neurosurrogate.utils import PLOTTER_REGISTRY
from neurosurrogate.utils.data_processing import _get_control_input
from neurosurrogate.utils.plots import plot_diff


@task
def single_eval(dataset_key, dataset_cfg, neuron_cfg, preprocessed_ds, surrogate_model):
    logger.info(f"{preprocessed_ds} started to process")
    data_type = dataset_cfg.get("data_type")

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
        params = neuron_cfg.get("params", {}) if neuron_cfg else {}
        calc_ThreeComp_internal(
            prediction,
            params.get("G_12"),
            params.get("G_23"),
        )

    logger.trace(prediction)
    return prediction


@task
def log_single_eval(dataset_key, dataset_cfg, surrogate_result, preprocessed_result):
    if surrogate_result is None:
        return

    data_type = dataset_cfg["data_type"]

    u = preprocessed_result["I_ext"].to_numpy()
    fig = plot_diff(u, preprocessed_result["vars"], surrogate_result["vars"])
    mlflow.log_figure(fig, f"compare/{data_type}/{dataset_key}.png")

    _debug_show_image(fig)

    plt.close(fig)
    fig = PLOTTER_REGISTRY[data_type](
        surrogate_result,
        surrogate=True,
    )

    mlflow.log_figure(
        fig,
        f"surrogate_result/{data_type}/{dataset_key}.png",
    )
    plt.close(fig)


def _debug_show_image(fig):
    with tempfile.TemporaryDirectory() as tmp_dir:
        import subprocess
        from pathlib import Path

        TMP = Path(tmp_dir) / "debug.png"
        fig.savefig(TMP)
        try:
            subprocess.run(["wezterm", "imgcat", TMP], check=False)
        except Exception:
            pass
