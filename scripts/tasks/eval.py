from datetime import datetime

import gokart
import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf

from neurosurrogate.config import (
    DATA_DIR,
    SURROGATE_DATA_DIR,
)

from .utils import CommonConfig


class EvalTask(gokart.TaskOnKart):
    def requires(self):
        from .train import PreProcessDataTask, TrainModelTask

        return {
            "preprocess_task": PreProcessDataTask(),
            "trainmodel_task": TrainModelTask(),
        }

    def run(self):
        """
        Load a registered model from MLflow and make a prediction.
        """
        loaded_data = self.load()
        conf = CommonConfig()

        slicer_time = hydra.utils.instantiate(
            OmegaConf.create(conf.eval_cfg_yaml).time_slice
        )
        datasets_cfg = OmegaConf.create(conf.datasets_cfg_yaml)
        path_dict = {}

        # PreProcessDataTask now returns the dictionary directly
        preprocessed_datasets = loaded_data["preprocess_task"]
        neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)
        for k, v in preprocessed_datasets.items():
            data_type = datasets_cfg[k].data_type

            neuron_cfg = neurons_cfg[data_type]

            logger.info(f"{v} started to process")
            ds = v.isel(time=slicer_time)
            if data_type == "hh":
                mode = "SingleComp"
                u = ds["I_ext"].to_numpy()
                if OmegaConf.create(conf.eval_cfg_yaml).onlyThreeComp is True:
                    logger.info(f"{k} is passed")
                    continue
            elif data_type == "hh3":
                mode = "ThreeComp"
                u = ds["I_ext"].to_numpy()
                if OmegaConf.create(conf.eval_cfg_yaml).direct is True:
                    logger.info("Using direct ThreeComp mode")

                    u_dic = neuron_cfg.transform.u
                    u = ds[u_dic.ind].sel(u_dic.sel).to_numpy()
                    mode = "SingleComp"

            input_data = {
                "init": ds["vars"][0],
                "dt": 0.01,
                "iter": len(ds["time"].to_numpy()),
                "u": u,
                "mode": mode,
            }
            logger.info(f"input:{input_data}")

            try:
                logger.critical(f"{k}")
                # TrainModelTask returns the surrogate model directly
                prediction = loaded_data["trainmodel_task"].predict(**input_data)
                logger.info(f"key:{k} prediction_result:{prediction}")
                if mode == "ThreeComp":
                    from neurosurrogate.dataset_utils._base import (
                        calc_ThreeComp_internal,
                    )

                    calc_ThreeComp_internal(prediction, neuron_cfg.params)

                logger.trace(prediction)

                file_path = SURROGATE_DATA_DIR / f"{datetime.now()}_{k}.npy"
                prediction.to_netcdf(file_path)
                path_dict[k] = file_path
            except ValueError as e:
                logger.error(f"Value Error: {e}")
        self.dump(path_dict)


class LogEvalTask(gokart.TaskOnKart):
    def requires(self):
        from .train import PreProcessDataTask

        return {"eval_task": EvalTask(), "preprocess_task": PreProcessDataTask()}

    def run(self):
        loaded_data = self.load()
        conf = CommonConfig()
        datasets_cfg = OmegaConf.create(conf.datasets_cfg_yaml)
        neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)
        with mlflow.start_run(run_id=conf.run_id):
            # EvalTask dumps {"path_dict": ...}
            for k, v in loaded_data["eval_task"].items():
                data_type = datasets_cfg[k].data_type
                # PreProcessDataTask dumps the dict directly
                preprocessed_result = loaded_data["preprocess_task"][k]

                with (
                    xr.open_dataset(v) as surrogate_result,
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

        self.dump(True)

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
