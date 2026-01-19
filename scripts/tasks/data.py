import hashlib
import random
import tempfile
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import OmegaConf
from prefect import task

from neurosurrogate.dataset_utils import PARAMS_REGISTRY, SIMULATOR_REGISTRY
from neurosurrogate.dataset_utils._base import preprocess_dataset
from neurosurrogate.utils import PLOTTER_REGISTRY

from .utils import fig_to_buff, recursive_to_dict


def compute_task_seed(dataset_cfg, neuron_cfg, base_seed) -> int:
    import json

    cfg_json = json.dumps(
        {
            "dataset": recursive_to_dict(dataset_cfg),
            "neuron": recursive_to_dict(neuron_cfg),
        },
        sort_keys=True,
    )

    hash_digest = hashlib.md5(cfg_json.encode()).hexdigest()
    # 16進数文字列を適切に処理
    return (base_seed + int(hash_digest, 16)) % (2**32)


def generate_single_dataset_key_fn(context, params):
    seed = params.get("task_seed")
    return f"{seed}"


@task(cache_key_fn=generate_single_dataset_key_fn, persist_result=True)
def generate_single_dataset(dataset_cfg, neuron_cfg, task_seed):
    """
    Simulates a neuron model based on configurations and preprocesses the result into a dataset.
    """
    # Configuration setup
    data_type = dataset_cfg["data_type"]
    params_dict = neuron_cfg.get("params")

    if params_dict is None:
        params = PARAMS_REGISTRY[data_type]()
    else:
        params = PARAMS_REGISTRY[data_type](**params_dict)

    # Set random seeds for reproducibility
    random.seed(task_seed)
    np.random.seed(task_seed)

    dataset_cfg_obj = OmegaConf.create(recursive_to_dict(dataset_cfg))
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_h5_path = Path(tmp_dir) / "sim_interim.h5"

        with h5py.File(temp_h5_path, "w") as fp:
            hydra.utils.instantiate(dataset_cfg_obj["current"], fp=fp, dt=params.DT)
            SIMULATOR_REGISTRY[data_type](fp=fp, params=params)
        # Preprocess the simulation data
        processed_dataset = preprocess_dataset(data_type, temp_h5_path, params_dict)
        # Load into memory to return
        processed_dataset.load()
        return processed_dataset


def log_single_dataset_key_fn(context, params):
    seed = params.get("task_seed")
    return f"log-{seed}"


@task(cache_key_fn=log_single_dataset_key_fn, persist_result=True)
def log_single_dataset(data_type, xr_data, task_seed):
    fig = PLOTTER_REGISTRY[data_type](xr_data)
    return fig_to_buff(fig)
