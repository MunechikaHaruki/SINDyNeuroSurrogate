import random

import hydra
import numpy as np
from prefect import task

from neurosurrogate.modeling import (
    SIMULATOR_REGISTRY,
    instantiate_OmegaConf_params,
)
from neurosurrogate.utils import PLOTTER_REGISTRY
from neurosurrogate.utils.data_processing import preprocess_dataset

from .utils import fig_to_buff, generate_complex_hash, log_plot_to_mlflow


def generate_single_dataset_key_fn(context, params):
    seed = params.get("task_seed")
    return f"{seed}"


@task(cache_key_fn=generate_single_dataset_key_fn, persist_result=True)
def generate_single_dataset(dataset_cfg, neuron_cfg, task_seed, DT):
    """
    Simulates a neuron model based on configurations and preprocesses the result into a dataset.
    """
    # Configuration setup
    data_type = dataset_cfg["data_type"]

    params = instantiate_OmegaConf_params(neuron_cfg, data_type=data_type)

    # Set random seeds for reproducibility
    random.seed(task_seed)
    np.random.seed(task_seed)

    i_ext = hydra.utils.instantiate(dataset_cfg["current"])
    results = SIMULATOR_REGISTRY[data_type](i_ext, params, DT)
    time_array = np.arange(len(i_ext)) * DT
    # Preprocess the simulation data
    processed_dataset = preprocess_dataset(
        data_type, i_ext, results, neuron_cfg, time_array
    )
    return processed_dataset


@task
def log_single_dataset(data_type, xr_data):
    fig = PLOTTER_REGISTRY[data_type](xr_data)
    return fig_to_buff(fig)


def generate_dataset_flow(dataset_key, cfg):
    dataset_cfg = cfg.datasets[dataset_key]
    data_type = dataset_cfg.data_type
    neuron_cfg = cfg.neurons.get(data_type)

    task_seed = generate_complex_hash(
        dataset_cfg,
        neuron_cfg,
        cfg.seed,
    )
    ds = generate_single_dataset(
        dataset_cfg=dataset_cfg,
        neuron_cfg=neuron_cfg,
        task_seed=int(task_seed, 16) % (2**32),
        DT=cfg.simulater_dt,
    )

    log_plot_to_mlflow(
        log_single_dataset(
            data_type=data_type,
            xr_data=ds,
        ),
        f"original/{data_type}/{dataset_key}.png",
    )
    return ds
