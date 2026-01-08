import random
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


class GenerateSingleDatasetTask(gokart.TaskOnKart):
    """
    Simulates a neuron model based on configurations and preprocesses the result into a dataset.
    """

    dataset_cfg_yaml = luigi.Parameter()
    neuron_cfg_yaml = luigi.Parameter()
    seed = luigi.IntParameter()

    def run(self):
        # Configuration setup
        dataset_cfg = OmegaConf.create(self.dataset_cfg_yaml)
        neuron_cfg = OmegaConf.create(self.neuron_cfg_yaml)
        params = hydra.utils.instantiate(neuron_cfg["params"])

        # Set random seeds for reproducibility
        self._set_random_seeds()

        # Prepare temporary H5 file path
        temp_h5_path = self._get_h5_file_path()

        # Run simulation
        self._run_simulation(temp_h5_path, dataset_cfg, neuron_cfg, params)

        # Preprocess the simulation data
        data_type = dataset_cfg["data_type"]
        processed_info = preprocess_dataset(data_type, temp_h5_path, params)

        self.dump(processed_info)

    def _set_random_seeds(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _get_h5_file_path(self) -> Path:
        target = self.make_target(
            "data.h5", processor=gokart.file_processor.BinaryFileProcessor()
        )
        path = Path(target.path())
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _run_simulation(self, file_path: Path, dataset_cfg, neuron_cfg, params):
        with h5py.File(file_path, "w") as fp:
            hydra.utils.instantiate(dataset_cfg["current"], fp=fp, dt=params.DT)
            hydra.utils.instantiate(neuron_cfg["simulator"], fp=fp, params=params)


class MakeDatasetTask(gokart.TaskOnKart):
    """データセットの生成と前処理を行うタスク"""

    seed = luigi.IntParameter()

    def requires(self):
        conf = CommonConfig()
        datasets = OmegaConf.create(conf.datasets_cfg_yaml)
        neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)
        return {
            name: GenerateSingleDatasetTask(
                dataset_cfg_yaml=OmegaConf.to_yaml(dataset_cfg),
                neuron_cfg_yaml=OmegaConf.to_yaml(neurons_cfg[dataset_cfg.data_type]),
                seed=self.seed + idx,
            )
            for idx, (name, dataset_cfg) in enumerate(datasets.items())
        }

    def run(self):
        self.dump({"path_dict": self.load()})


class LogSingleDatasetTask(gokart.TaskOnKart):
    dataset_name = luigi.Parameter()
    dataset_cfg_yaml = luigi.Parameter()
    neuron_cfg_yaml = luigi.Parameter()
    seed = luigi.IntParameter()
    run_id = luigi.Parameter()

    def requires(self):
        return GenerateSingleDatasetTask(
            dataset_cfg_yaml=self.dataset_cfg_yaml,
            neuron_cfg_yaml=self.neuron_cfg_yaml,
            seed=self.seed,
        )

    def run(self):
        dataset_cfg = OmegaConf.create(self.dataset_cfg_yaml)
        neuron_cfg = OmegaConf.create(self.neuron_cfg_yaml)
        file_path = self.load()

        with mlflow.start_run(run_id=self.run_id):
            with xr.open_dataset(Path(file_path)) as xr_data:
                fig = hydra.utils.instantiate(neuron_cfg.plot, xr=xr_data)
                mlflow.log_figure(
                    fig, f"original/{dataset_cfg.data_type}/{self.dataset_name}.png"
                )
                plt.close(fig)
        logger.info(f"Logged dataset: {self.dataset_name}")
        self.dump(self.dataset_name)


class LogMakeDatasetTask(gokart.TaskOnKart):
    def run(self):
        conf = CommonConfig()
        datasets = OmegaConf.create(conf.datasets_cfg_yaml)
        neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)

        run_id = getattr(conf, "run_id", None)
        if run_id is None:
            with mlflow.start_run() as run:
                run_id = run.info.run_id

        tasks = []
        seed = CommonConfig().seed
        for idx, (name, dataset_cfg) in enumerate(datasets.items()):
            tasks.append(
                LogSingleDatasetTask(
                    dataset_name=name,
                    dataset_cfg_yaml=OmegaConf.to_yaml(dataset_cfg),
                    neuron_cfg_yaml=OmegaConf.to_yaml(
                        neurons_cfg[dataset_cfg.data_type]
                    ),
                    seed=seed + idx,
                    run_id=run_id,
                )
            )
        yield tasks
        self.dump(True)

