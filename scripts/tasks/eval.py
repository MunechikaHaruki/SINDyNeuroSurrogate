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

from .utils import CommonConfig


class EvalTask(gokart.TaskOnKart):
    preprocess_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.preprocess_task

    def run(self):
        """
        Load a registered model from MLflow and make a prediction.
        """
        conf = CommonConfig()
        with mlflow.start_run(run_id=self.load()["run_id"]):
            model = mlflow.pyfunc.load_model(f"runs:/{self.load()['run_id']}/model")
        slicer_time = hydra.utils.instantiate(
            OmegaConf.create(conf.eval_cfg_yaml).time_slice
        )
        datasets_cfg = OmegaConf.create(conf.datasets_cfg_yaml)
        path_dict = {}
        for k, v in self.load()["path_dict"].items():
            logger.info(f"{v} started to process")
            ds = xr.open_dataset(v).isel(time=slicer_time)
            if datasets_cfg[k].data_type == "hh":
                mode = "SingleComp"
                u = ds["I_ext"].to_numpy()
                if OmegaConf.create(conf.eval_cfg_yaml).onlyThreeComp is True:
                    logger.info(f"{k} is passed")
                    continue
            elif datasets_cfg[k].data_type == "hh3":
                mode = "ThreeComp"
                u = ds["I_ext"].to_numpy()
                if OmegaConf.create(conf.eval_cfg_yaml).direct is True:
                    logger.info("Using direct ThreeComp mode")
                    neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)
                    u_dic = neurons_cfg[datasets_cfg[k].data_type].transform.u
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
    eval_task = gokart.TaskInstanceParameter()
    preprocess_task = gokart.TaskInstanceParameter()

    def requires(self):
        return {"eval_task": self.eval_task, "preprocess_task": self.preprocess_task}

    def run(self):
        loaded_data = self.load()
        conf = CommonConfig()
        datasets_cfg = OmegaConf.create(conf.datasets_cfg_yaml)
        neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)
        run_id = loaded_data["eval_task"]["run_id"]
        with mlflow.start_run(run_id=run_id):
            for k, v in loaded_data["eval_task"]["path_dict"].items():
                data_type = datasets_cfg[k].data_type
                preprocessed_path = loaded_data["preprocess_task"]["path_dict"][k]

                with (
                    xr.open_dataset(v) as surrogate_result,
                    xr.open_dataset(preprocessed_path) as preprocessed_result,
                ):
                    u = preprocessed_result["I_ext"].to_numpy()
                    fig = self.plot_diff(
                        u, preprocessed_result["vars"], surrogate_result["vars"]
                    )
                    mlflow.log_figure(fig, f"compare/{data_type}/{k}.png")

                    self._debug_show_image(fig)

                    plt.close(fig)
                    fig = hydra.utils.instantiate(
                        neurons_cfg[datasets_cfg[k].data_type].plot,
                        xr=surrogate_result,
                        surrogate=True,
                    )

                    mlflow.log_figure(
                        fig,
                        f"surrogate_result/{data_type}/{k}.png",
                    )
                    plt.close(fig)

        self.dump({"run_id": run_id})

    def plot_diff(self, u: np.ndarray, original: xr.DataArray, surrogate: xr.DataArray):
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

    def _debug_show_image(self, fig):
        import subprocess

        TMP = DATA_DIR / "debug.png"
        fig.savefig(TMP)
        subprocess.run(["wezterm", "imgcat", TMP])
