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

from neurosurrogate.dataset_utils._base import preprocess_dataset

from .utils import CommonConfig


class NetCDFProcessor(gokart.file_processor.FileProcessor):
    def format(self):
        return luigi.format.Nop

    def load(self, input_file):
        with xr.open_dataset(input_file, engine="h5netcdf") as ds:
            return ds.load()

    def dump(self, obj, output_file):
        obj.to_netcdf(output_file.name, engine="h5netcdf")


class GenerateSingleDatasetTask(gokart.TaskOnKart):
    """
    Simulates a neuron model based on configurations and preprocesses the result into a dataset.
    """

    dataset_cfg_yaml = luigi.Parameter()
    neuron_cfg_yaml = luigi.Parameter()
    seed = luigi.IntParameter(default=CommonConfig().seed)

    def output(self):
        return self.make_target("dataset.nc", processor=NetCDFProcessor())

    def run(self):
        # Configuration setup
        dataset_cfg = OmegaConf.create(self.dataset_cfg_yaml)
        neuron_cfg = OmegaConf.create(self.neuron_cfg_yaml)
        params = hydra.utils.instantiate(neuron_cfg["params"])

        # Set random seeds for reproducibility
        self._set_random_seeds()

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_h5_path = Path(tmp_dir) / "sim_interim.h5"
            self._run_simulation(temp_h5_path, dataset_cfg, neuron_cfg, params)
            # Preprocess the simulation data
            data_type = dataset_cfg["data_type"]
            processed_dataset = preprocess_dataset(data_type, temp_h5_path, params)
            self.dump(processed_dataset)

    def _set_random_seeds(self):
        hash_digest = hashlib.md5(self.dataset_cfg_yaml.encode()).hexdigest()
        task_seed = (self.seed + int(hash_digest, 16)) % 100000000
        logger.debug(task_seed)
        random.seed(task_seed)
        np.random.seed(task_seed)

    def _run_simulation(self, file_path: Path, dataset_cfg, neuron_cfg, params):
        with h5py.File(file_path, "w") as fp:
            hydra.utils.instantiate(dataset_cfg["current"], fp=fp, dt=params.DT)
            hydra.utils.instantiate(neuron_cfg["simulator"], fp=fp, params=params)


class MakeDatasetTask(gokart.TaskOnKart):
    """データセットの生成と前処理を行うタスク"""

    def requires(self):
        conf = CommonConfig()
        datasets = OmegaConf.create(conf.datasets_cfg_yaml)
        neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)
        return {
            name: GenerateSingleDatasetTask(
                dataset_cfg_yaml=OmegaConf.to_yaml(dataset_cfg),
                neuron_cfg_yaml=OmegaConf.to_yaml(neurons_cfg[dataset_cfg.data_type]),
            )
            for name, dataset_cfg in datasets.items()
        }

    def run(self):
        targets = self.input()
        dataset_path_mapping = {name: target.path() for name, target in targets.items()}
        self.dump(dataset_path_mapping)


class LogSingleDatasetTask(gokart.TaskOnKart):
    dataset_name = luigi.Parameter()
    dataset_cfg_yaml = luigi.Parameter()
    neuron_cfg_yaml = luigi.Parameter()
    run_id = luigi.Parameter()

    def requires(self):
        return GenerateSingleDatasetTask(
            dataset_cfg_yaml=self.dataset_cfg_yaml,
            neuron_cfg_yaml=self.neuron_cfg_yaml,
        )

    def run(self):
        dataset_cfg = OmegaConf.create(self.dataset_cfg_yaml)
        neuron_cfg = OmegaConf.create(self.neuron_cfg_yaml)

        with mlflow.start_run(run_id=self.run_id):
            xr_data = self.load()
            fig = hydra.utils.instantiate(neuron_cfg.plot, xr=xr_data)
            mlflow.log_figure(
                fig, f"original/{dataset_cfg.data_type}/{self.dataset_name}.png"
            )
            plt.close(fig)
        logger.info(f"Logged dataset: {self.dataset_name}")
        self.dump(True)


class LogMakeDatasetTask(gokart.TaskOnKart):
    def run(self):
        conf = CommonConfig()
        datasets = OmegaConf.create(conf.datasets_cfg_yaml)
        neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)

        tasks = []
        for name, dataset_cfg in datasets.items():
            tasks.append(
                LogSingleDatasetTask(
                    dataset_name=name,
                    dataset_cfg_yaml=OmegaConf.to_yaml(dataset_cfg),
                    neuron_cfg_yaml=OmegaConf.to_yaml(
                        neurons_cfg[dataset_cfg.data_type]
                    ),
                    run_id=conf.run_id,
                )
            )
        yield tasks
        self.dump(True)
