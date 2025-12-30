import os
import random
import time
from datetime import datetime

import gokart
import h5py
import hydra
import luigi
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import xarray as xr
from loguru import logger
from neurosurrogate.dataset_utils.hh.hh_simulator import hh_simulate, threecomp_simulate
from neurosurrogate.dataset_utils.traub.traub_simulator import traub_simulate
from omegaconf import OmegaConf

from neurosurrogate.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from neurosurrogate.dataset import preprocess_dataset
from neurosurrogate.plots import plot_3comp_hh, plot_hh


class MakeDatasetTask(gokart.TaskOnKart):
    """データセットの生成と前処理を行うタスク"""

    datasets_cfg_yaml = luigi.Parameter()
    neurons_cfg_yaml = luigi.Parameter()
    experiment_name = luigi.Parameter()
    seed = luigi.IntParameter()

    def run(self):
        random.seed(self.seed)
        path_dict = {}
        neuron_cfg = OmegaConf.create(self.neurons_cfg_yaml)

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run.info.run_id}")

        for name, dataset_cfg in OmegaConf.create(self.datasets_cfg_yaml).items():
            file_name = f"{datetime.now()}_{name}.h5"
            params = hydra.utils.instantiate(neuron_cfg[dataset_cfg.data_type].params)
            output_path = RAW_DATA_DIR / dataset_cfg.data_type / file_name

            with h5py.File(output_path, "w") as fp:
                hydra.utils.instantiate(dataset_cfg.current, fp=fp, dt=params.DT)
            # Simulation
            SIMULATORS = {
                "hh": hh_simulate,
                "hh3": threecomp_simulate,
                "traub": traub_simulate,
            }
            with h5py.File(output_path, "a") as fp:
                start_time = time.perf_counter()
                SIMULATORS[dataset_cfg.data_type](fp, params)
                end_time = time.perf_counter()
                logger.info(f"Simulation time: {end_time - start_time:.4f}[s]")
            logger.success(f"Dataset generation complete: {output_path}")
            path_dict[name] = preprocess_dataset(
                dataset_cfg.data_type, file_name, params
            )

        self.dump({"path_dict": path_dict, "run_id": run_id})


class LogMakeDatasetTask(gokart.TaskOnKart):
    datasets_cfg_yaml = luigi.Parameter()
    dataset_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.dataset_task

    def run(self):
        path_dict = self.load()["path_dict"]
        with mlflow.start_run(run_id=self.load()["run_id"]):
            for name, dataset_cfg in OmegaConf.create(self.datasets_cfg_yaml).items():
                PLOTTERS = {
                    "hh": plot_hh,
                    "hh3": plot_3comp_hh,
                }
                xr_data = xr.open_dataset(path_dict[name])
                fig = PLOTTERS[dataset_cfg.data_type](xr_data)
                mlflow.log_figure(fig, f"oridginal/{name}.png")
                plt.close(fig)
            logger.info(f"Generated and preprocessed dataset: {name}")
        self.dump(True)


GATE_VAR_SLICE = slice(1, 4, None)
V_VAR_SLICE = slice(0, 1, None)


class PreProcessTask(gokart.TaskOnKart):
    train_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.train_task

    def run(self):
        preprocessed_path_dict = {}
        # preprocess data
        for k, v in self.load()["path_dict"].items():
            xr_data = xr.open_dataset(v)
            xr_gate = xr_data["vars"].to_numpy()[:, GATE_VAR_SLICE]
            transformed_gate = self.load()["preprocessor"].transform(xr_gate)
            V_data = xr_data["vars"][:, V_VAR_SLICE].to_numpy().reshape(-1, 1)
            new_vars = np.concatenate((V_data, transformed_gate), axis=1)
            new_feature_names = ["V"] + [
                f"latent{i + 1}" for i in range(transformed_gate.shape[1])
            ]
            transformed_xr = xr_data.copy().drop_vars("vars").drop_vars("features")
            transformed_xr["vars"] = xr.DataArray(
                new_vars,
                coords={
                    "time": xr_data.coords["time"],
                    "features": new_feature_names,  # 新しい次元と座標
                },
                dims=["time", "features"],  # 新しい次元名
            )
            logger.info(f"Transformed xr dataset: {k}")
            logger.info(transformed_gate.__repr__())
            preprocessed_path_dict[k] = PROCESSED_DATA_DIR / os.path.basename(v)
            transformed_xr.to_netcdf(preprocessed_path_dict[k])
        self.dump(
            {"path_dict": preprocessed_path_dict, "run_id": self.load()["run_id"]}
        )


class LogPreprocessDataTask(gokart.TaskOnKart):
    preprocess_task = gokart.TaskInstanceParameter()
    datasets_cfg_yaml = luigi.Parameter()
    neuron_cfg_yaml = luigi.Parameter()

    def requires(self):
        return self.preprocess_task

    def run(self):
        datasets_cfg = OmegaConf.create(self.datasets_cfg_yaml)
        neurons_cfg = OmegaConf.create(self.neuron_cfg_yaml)

        with mlflow.start_run(run_id=self.load()["run_id"]):
            for k, v in self.load()["path_dict"].items():
                xr_data = xr.load_dataset(v)
                dataset_type = datasets_cfg[k].data_type
                u_dic = neurons_cfg[dataset_type].transform.u
                data = xr_data["vars"]
                external_input = xr_data[u_dic.ind].sel(u_dic.sel)
                num_features = len(data.features.values)

                fig, axs = plt.subplots(
                    1 + num_features,
                    1,
                    figsize=(10, 4 * (1 + num_features)),
                    sharex=True,
                )

                axs[0].plot(external_input.time, external_input, label="I_ext(t)")
                axs[0].set_ylabel("I_ext(t)")
                axs[0].legend()

                for i, feature in enumerate(data.features.values):
                    axs[i + 1].plot(
                        data.time, data.sel(features=feature), label=feature
                    )
                    axs[i + 1].set_ylabel(feature)
                    axs[i + 1].legend()
                axs[-1].set_xlabel("Time step")

                mlflow.log_figure(
                    fig,
                    f"preprocessed/{k}.png",
                )
                plt.close(fig)
        self.dump(True)
