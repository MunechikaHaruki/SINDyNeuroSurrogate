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
    dataset_cfg_yaml = luigi.Parameter()
    neuron_cfg_yaml = luigi.Parameter()
    seed = luigi.IntParameter()

    def run(self):
        dataset_cfg = OmegaConf.create(self.dataset_cfg_yaml)
        data_type = dataset_cfg["data_type"]
        neuron_cfg = OmegaConf.create(self.neuron_cfg_yaml)
        params = hydra.utils.instantiate(neuron_cfg["params"])
        random.seed(self.seed)
        np.random.seed(self.seed)

        temp_h5 = Path(
            self.make_target(
                "data.h5", processor=gokart.file_processor.BinaryFileProcessor()
            ).path()
        )
        temp_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(temp_h5, "w") as fp:
            hydra.utils.instantiate(dataset_cfg["current"], fp=fp, dt=params.DT)
            hydra.utils.instantiate(neuron_cfg["simulator"], fp=fp, params=params)
        processed_info = preprocess_dataset(data_type, temp_h5, params)
        self.dump(processed_info)


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


class LogMakeDatasetTask(gokart.TaskOnKart):
    def requires(self):
        return MakeDatasetTask(seed=CommonConfig().seed)

    def run(self):
        conf = CommonConfig()
        loaded_data = self.load()
        neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)
        with mlflow.start_run(run_id=conf.run_id):
            for name, dataset_cfg in OmegaConf.create(conf.datasets_cfg_yaml).items():
                with xr.open_dataset(loaded_data["path_dict"][name]) as xr_data:
                    fig = hydra.utils.instantiate(
                        neurons_cfg[dataset_cfg.data_type].plot, xr=xr_data
                    )
                    mlflow.log_figure(
                        fig, f"original/{dataset_cfg.data_type}/{name}.png"
                    )
                    plt.close(fig)
                logger.info(f"Logged dataset: {name}")
        self.dump(True)
