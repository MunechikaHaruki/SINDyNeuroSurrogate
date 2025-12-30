import subprocess
from datetime import datetime

import gokart
import hydra
import luigi
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf

from neurosurrogate.config import (
    DATA_DIR,
    SURROGATE_DATA_DIR,
)
from neurosurrogate.plots import plot_3comp_hh, plot_hh


class EvalTask(gokart.TaskOnKart):
    preprocess_task = gokart.TaskInstanceParameter()
    eval_cfg_yaml = luigi.Parameter()
    neuron_cfg_yaml = luigi.Parameter()
    datasets_cfg_yaml = luigi.Parameter()

    def requires(self):
        return self.preprocess_task

    def run(self):
        """
        Load a registered model from MLflow and make a prediction.
        """
        with mlflow.start_run(run_id=self.load()["run_id"]):
            model = mlflow.pyfunc.load_model(f"runs:/{self.load()['run_id']}/model")
        slicer_time = hydra.utils.instantiate(
            OmegaConf.create(self.eval_cfg_yaml).time_slice
        )
        datasets_cfg = OmegaConf.create(self.datasets_cfg_yaml)
        path_dict = {}
        for k, v in self.load()["path_dict"].items():
            logger.info(f"{v} started to process")
            ds = xr.open_dataset(v).isel(time=slicer_time)
            if datasets_cfg[k].data_type == "hh":
                mode = "SingleComp"
                u = ds["I_ext"].to_numpy()
                if OmegaConf.create(self.eval_cfg_yaml).onlyThreeComp is True:
                    logger.info(f"{k} is passed")
                    continue
            elif datasets_cfg[k].data_type == "hh3":
                mode = "ThreeComp"
                u = ds["I_ext"].to_numpy()
                if OmegaConf.create(self.eval_cfg_yaml).direct is True:
                    logger.info("Using direct ThreeComp mode")
                    neuron_cfg = OmegaConf.create(self.neuron_cfg_yaml)
                    u_dic = neuron_cfg[datasets_cfg[k].data_type].transform.u
                    u = ds[u_dic.ind].sel(u_dic.sel).to_numpy()
                    # u = ds["I_ext"].to_numpy() # 11/4のデータは,hh3,SingleComp,I_extでの予測　間違い
                    mode = "SingleComp"

            input_data = pd.DataFrame(
                {
                    "init": [ds["vars"][0]],
                    "dt": [0.01 * slicer_time.step],
                    "iter": [len(ds["time"].to_numpy())],
                    "u": [u],
                    "mode": [mode],
                }
            )
            logger.info(f"input:{input_data}")

            try:
                logger.critical(f"{k}")
                prediction = model.predict(input_data)
                logger.info(f"key:{k} prediction_result:{prediction}")
                if mode == "ThreeComp":
                    I_pre = 1 * (
                        prediction["vars"].sel(features="V_pre")
                        - prediction["vars"].sel(features="V")
                    )
                    I_post = 0.7 * (
                        prediction["vars"].sel(features="V")
                        - prediction["vars"].sel(features="V_post")
                    )
                    I_soma = I_pre - I_post
                    prediction["I_internal"] = xr.concat(
                        [I_pre, I_post, I_soma], dim="direction"
                    ).assign_coords(direction=["pre", "post", "soma"])
                logger.trace(prediction)

                file_path = SURROGATE_DATA_DIR / f"{datetime.now()}_{k}.npy"
                prediction.to_netcdf(file_path)
                path_dict[k] = file_path
            except ValueError as e:
                logger.error(f"Value Error: {e}")
        self.dump({"run_id": self.load()["run_id"], "path_dict": path_dict})


class LogEvalTask(gokart.TaskOnKart):
    datasets_cfg_yaml = luigi.Parameter()
    eval_task = gokart.TaskInstanceParameter()
    preprocess_task = gokart.TaskInstanceParameter()

    def requires(self):
        return {"eval_task": self.eval_task, "preprocess_task": self.preprocess_task}

    def run(self):
        def plot_diff(u: np.ndarray, original: xr.DataArray, surrogate: xr.DataArray):
            num_features = len(original.features.values)

            fig, axs = plt.subplots(
                1 + 2 * num_features,
                1,
                figsize=(10, 4 * (1 + num_features)),
                sharex=False,
            )

            # plot external_input (I_ext)
            axs[0].plot(u, label="I_ext(t)", color="gold")
            axs[0].set_ylabel("I_ext(t)")
            axs[0].legend()

            # 各 feature についてループ
            for i, feature in enumerate(original.features.values):
                # 1. 元のデータをプロット (引数 'oridginal' から)
                axs[2 * i + 1].plot(
                    original.time,
                    original.sel(features=feature),
                    color="blue",
                    label=f"Original {feature}",
                )
                axs[2 * i + 1].set_ylabel(feature)
                axs[2 * i + 1].legend()

                # 2. サロゲートモデルのデータをプロット (引数 'surrogate' から)
                #    surrogate も 'time' と 'features' の座標を持つと仮定
                axs[2 * i + 2].plot(
                    surrogate.time,
                    surrogate.sel(features=feature),
                    color="red",
                    label=f"Surrogate {feature}",
                )
                axs[2 * i + 2].set_ylabel(f"Surrogate {feature}")
                axs[2 * i + 2].legend()

            axs[-1].set_xlabel("Time step")
            fig.tight_layout()  # レイアウトを自動調整
            return fig

        datasets_cfg = OmegaConf.create(self.datasets_cfg_yaml)
        with mlflow.start_run(run_id=self.load()["eval_task"]["run_id"]):
            for k, v in self.load()["eval_task"]["path_dict"].items():
                if datasets_cfg[k].data_type == "hh":
                    plot_surrogate = plot_hh
                elif datasets_cfg[k].data_type == "hh3":
                    plot_surrogate = plot_3comp_hh
                surrogate_result = xr.open_dataset(v)
                preprocessed_result = xr.open_dataset(
                    self.load()["preprocess_task"]["path_dict"][k]
                )
                u = preprocessed_result["I_ext"].to_numpy()
                fig = plot_diff(
                    u, preprocessed_result["vars"], surrogate_result["vars"]
                )
                mlflow.log_figure(fig, f"compare/{k}.png")
                TMP = DATA_DIR / "show.png"
                fig.savefig(TMP)
                subprocess.run(["wezterm", "imgcat", TMP])

                mlflow.log_figure(
                    plot_surrogate(surrogate_result, surrogate=True),
                    f"surrogate_result/{k}.png",
                )
        self.dump(True)
