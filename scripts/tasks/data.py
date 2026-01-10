import hashlib
import random
import tempfile
from pathlib import Path

import gokart
import gokart.file_processor
import h5py
import hydra
import luigi
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf

from neurosurrogate.dataset_utils import PARAMS_REGISTRY, SIMULATOR_REGISTRY
from neurosurrogate.dataset_utils._base import preprocess_dataset

from .utils import CommonConfig, recursive_to_dict


class NetCDFProcessor(gokart.file_processor.FileProcessor):
    def format(self):
        return luigi.format.Nop

    def load(self, input_file):
        with xr.open_dataset(input_file, engine="h5netcdf") as ds:
            return ds.load()

    def dump(self, obj, output_file):
        obj.to_netcdf(output_file.name, engine="h5netcdf")
        logger.debug(f"{output_file.name} is dumped")


class GenerateSingleDatasetTask(gokart.TaskOnKart):
    """
    Simulates a neuron model based on configurations and preprocesses the result into a dataset.
    """

    dataset_cfg = luigi.DictParameter()
    neuron_cfg = luigi.DictParameter()
    seed = luigi.IntParameter(default=CommonConfig().seed)

    def output(self):
        return self.make_target("dataset.nc", processor=NetCDFProcessor())

    def run(self):
        # Configuration setup
        data_type = self.dataset_cfg["data_type"]
        params_dict = self.neuron_cfg["params"]

        if params_dict is None:
            params = PARAMS_REGISTRY[data_type]()
        else:
            params = PARAMS_REGISTRY[data_type](**params_dict)

        # Set random seeds for reproducibility
        self._set_random_seeds()

        dataset_cfg = OmegaConf.create(recursive_to_dict(self.dataset_cfg))
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_h5_path = Path(tmp_dir) / "sim_interim.h5"

            with h5py.File(temp_h5_path, "w") as fp:
                hydra.utils.instantiate(dataset_cfg["current"], fp=fp, dt=params.DT)
                SIMULATOR_REGISTRY[data_type](fp=fp, params=params)
            # Preprocess the simulation data
            processed_dataset = preprocess_dataset(data_type, temp_h5_path, params)
            self.dump(processed_dataset)

    def _set_random_seeds(self):
        import json

        cfg_json = json.dumps(
            {
                "dataset": recursive_to_dict(self.dataset_cfg),
                "neuron": recursive_to_dict(self.neuron_cfg),
            },
            sort_keys=True,
        )

        hash_digest = hashlib.md5(cfg_json.encode()).hexdigest()
        # 16進数文字列を適切に処理
        task_seed = (self.seed + int(hash_digest, 16)) % (2**32)

        random.seed(task_seed)
        np.random.seed(task_seed)
        logger.info(f"Seed set to: {task_seed}")


class MakeDatasetTask(gokart.TaskOnKart):
    """データセットの生成と前処理を行うタスク"""

    def requires(self):
        conf = CommonConfig()
        neurons_cfg = OmegaConf.create(recursive_to_dict(conf.neurons_dict))
        return {
            name: GenerateSingleDatasetTask(
                dataset_cfg=conf.datasets_dict[name],
                neuron_cfg=neurons_cfg[conf.datasets_dict[name]["data_type"]],
            )
            for name in conf.datasets_dict.keys()
        }

    def run(self):
        targets = self.input()
        self.dump(targets)


class LogSingleDatasetTask(gokart.TaskOnKart):
    dataset_name = luigi.Parameter()
    dataset_cfg = luigi.DictParameter()
    neuron_cfg = luigi.DictParameter()
    run_id = luigi.Parameter()

    def requires(self):
        return GenerateSingleDatasetTask(
            dataset_cfg=self.dataset_cfg,
            neuron_cfg=self.neuron_cfg,
        )

    def run(self):
        neuron_cfg = OmegaConf.create(recursive_to_dict(self.neuron_cfg))
        data_type = self.dataset_cfg["data_type"]
        with mlflow.start_run(run_id=self.run_id):
            xr_data = self.load()
            fig = hydra.utils.instantiate(neuron_cfg.plot, xr=xr_data)
            mlflow.log_figure(fig, f"original/{data_type}/{self.dataset_name}.png")
            plt.close(fig)
        logger.info(f"Logged dataset: {self.dataset_name}")
        self.dump(True)


class LogMakeDatasetTask(gokart.TaskOnKart):
    def requires(self):
        conf = CommonConfig()

        return {
            name: LogSingleDatasetTask(
                dataset_name=name,
                dataset_cfg=conf.datasets_dict[name],
                neuron_cfg=conf.neurons_dict[conf.datasets_dict[name]["data_type"]],
                run_id=conf.run_id,
            )
            for name in conf.datasets_dict.keys()
        }

    def run(self):
        self.dump(True)
